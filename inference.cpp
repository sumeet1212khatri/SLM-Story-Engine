/*
 * OPTIMIZED SLM 50M INFERENCE ENGINE
 * Target: i3 11th Gen | Windows 11 | 8GB RAM
 * OpenMP Parallel + AVX2 Auto Vectorized
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <immintrin.h> // REQUIRED FOR AVX2 SIMD

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Config & Structures
// ---------------------------------------------------------------------------

typedef struct {
    int n_layer;
    int n_head;
    int n_embd;
    int block_size;
    int vocab_size;
} Config;

typedef struct {
    float* wte; float* wpe;
    float** ln1_w; float** ln1_b;
    float** c_attn_w; float** c_attn_b;
    float** c_proj_w; float** c_proj_b;
    float** ln2_w; float** ln2_b;
    float** fc_w; float** fc_b;
    float** mlp_proj_w; float** mlp_proj_b;
    float* ln_f_w; float* ln_f_b;
    float* lm_head_w;
} Weights;

typedef struct { float* k_cache; float* v_cache; } KVCache;

static Config cfg;
static Weights W;
static float* model_data_buffer = NULL;

// ---------------------------------------------------------------------------
// Math Kernels
// ---------------------------------------------------------------------------

static void layer_norm(float* out, const float* x, const float* w, const float* b, int size) {
    float mean = 0.0f, var = 0.0f;

    for (int i = 0; i < size; i++) mean += x[i];
    mean /= size;

    for (int i = 0; i < size; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= size;

    float scale = 1.0f / sqrtf(var + 1e-5f);

    for (int i = 0; i < size; i++)
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
}

// OpenMP + AVX2 + FMA parallelized matmul
static void matmul_vec(float* out, const float* mat, const float* x, int M, int K) {

#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        const float* row = mat + (long long)i * K;
        
        // Initialize a 256-bit vector with all zeros
        __m256 sum_vec = _mm256_setzero_ps();
        
        int j = 0;
        // Process 8 floats at a time
        for (; j <= K - 8; j += 8) {
            // Load 8 floats from the matrix row and the input vector
            __m256 m_val = _mm256_loadu_ps(&row[j]);
            __m256 x_val = _mm256_loadu_ps(&x[j]);
            
            // FMA (Fused Multiply-Add): sum_vec += m_val * x_val
            sum_vec = _mm256_fmadd_ps(m_val, x_val, sum_vec);
        }
        
        // Extract the 8 floats back out and sum them horizontally
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] + 
                    sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

        // Handle any leftover elements if K is not a multiple of 8
        for (; j < K; j++) {
            sum += row[j] * x[j];
        }

        out[i] = sum;
    }
}

static void add_bias(float* x, const float* b, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        x[i] += b[i];
}

static void residual_add(float* x, const float* y, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        x[i] += y[i];
}

static void gelu_inplace(float* x, int N) {
    const float c = 0.7978845608f;

#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float v = x[i];
        float t = tanhf(c * (v + 0.044715f * v * v * v));
        x[i] = 0.5f * v * (1.0f + t);
    }
}

static void softmax_inplace(float* x, int N) {

    float max_val = x[0];
    for (int i = 1; i < N; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < N; i++)
        x[i] /= sum;
}

// ---------------------------------------------------------------------------
// Transformer Forward
// ---------------------------------------------------------------------------

static void forward(
    int token_id,
    int pos,
    KVCache* kv,
    float* x,
    float* buf,
    float* qkv_buf,
    float* attn_buf,
    float* ff_buf,
    float* logits
) {
    const int C = cfg.n_embd;
    const int H = cfg.n_head;
    const int hs = C / H;

    float* content_row = W.wte + (long long)token_id * C;
    float* pos_row = W.wpe + (long long)pos * C;

#pragma omp parallel for
    for (int i = 0; i < C; i++)
        x[i] = content_row[i] + pos_row[i];

    for (int l = 0; l < cfg.n_layer; l++) {

        layer_norm(buf, x, W.ln1_w[l], W.ln1_b[l], C);

        matmul_vec(qkv_buf, W.c_attn_w[l], buf, 3 * C, C);
        add_bias(qkv_buf, W.c_attn_b[l], 3 * C);

        float* q = qkv_buf;
        float* k = qkv_buf + C;
        float* v = qkv_buf + 2 * C;

        float* k_cache = kv->k_cache + (long long)l * cfg.block_size * C;
        float* v_cache = kv->v_cache + (long long)l * cfg.block_size * C;

        memcpy(k_cache + (long long)pos * C, k, C * sizeof(float));
        memcpy(v_cache + (long long)pos * C, v, C * sizeof(float));

#pragma omp parallel for
        for (int h = 0; h < H; h++) {

            float* q_h = q + h * hs;
            float scale = 1.0f / sqrtf((float)hs);
            
            // Give each thread its own slice of the attention buffer
            float* local_attn = attn_buf + h * cfg.block_size;

            for (int t = 0; t <= pos; t++) {
                float* k_h = k_cache + (long long)t * C + h * hs;
                float dot = 0.0f;
                for (int d = 0; d < hs; d++)
                    dot += q_h[d] * k_h[d];

                local_attn[t] = dot * scale;
            }

            softmax_inplace(local_attn, pos + 1);

            float* out_h = buf + h * hs;
            memset(out_h, 0, hs * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                float* v_h = v_cache + (long long)t * C + h * hs;
                float a = local_attn[t];
                for (int d = 0; d < hs; d++)
                    out_h[d] += a * v_h[d];
            }
        }

        float* attn_out = qkv_buf;
        matmul_vec(attn_out, W.c_proj_w[l], buf, C, C);
        add_bias(attn_out, W.c_proj_b[l], C);
        residual_add(x, attn_out, C);

        layer_norm(buf, x, W.ln2_w[l], W.ln2_b[l], C);

        matmul_vec(ff_buf, W.fc_w[l], buf, 4 * C, C);
        add_bias(ff_buf, W.fc_b[l], 4 * C);
        gelu_inplace(ff_buf, 4 * C);

        matmul_vec(buf, W.mlp_proj_w[l], ff_buf, C, 4 * C);
        add_bias(buf, W.mlp_proj_b[l], C);
        residual_add(x, buf, C);
    }

    layer_norm(buf, x, W.ln_f_w, W.ln_f_b, C);
    matmul_vec(logits, W.lm_head_w, buf, cfg.vocab_size, C);
}

// ---------------------------------------------------------------------------
// Weight Mapping
// ---------------------------------------------------------------------------

static void map_weights(float* data) {

    float* ptr = data;
    const int C = cfg.n_embd;
    const int L = cfg.n_layer;

    W.wte = ptr; ptr += (long long)cfg.vocab_size * C;
    W.wpe = ptr; ptr += (long long)cfg.block_size * C;

    W.ln1_w = (float**)malloc(L * sizeof(float*));
    W.ln1_b = (float**)malloc(L * sizeof(float*));
    W.c_attn_w = (float**)malloc(L * sizeof(float*));
    W.c_attn_b = (float**)malloc(L * sizeof(float*));
    W.c_proj_w = (float**)malloc(L * sizeof(float*));
    W.c_proj_b = (float**)malloc(L * sizeof(float*));
    W.ln2_w = (float**)malloc(L * sizeof(float*));
    W.ln2_b = (float**)malloc(L * sizeof(float*));
    W.fc_w = (float**)malloc(L * sizeof(float*));
    W.fc_b = (float**)malloc(L * sizeof(float*));
    W.mlp_proj_w = (float**)malloc(L * sizeof(float*));
    W.mlp_proj_b = (float**)malloc(L * sizeof(float*));

    for (int l = 0; l < L; l++) {
        W.ln1_w[l] = ptr; ptr += C;
        W.ln1_b[l] = ptr; ptr += C;

        W.c_attn_w[l] = ptr; ptr += 3LL * C * C;
        W.c_attn_b[l] = ptr; ptr += 3LL * C;

        W.c_proj_w[l] = ptr; ptr += 1LL * C * C;
        W.c_proj_b[l] = ptr; ptr += C;

        W.ln2_w[l] = ptr; ptr += C;
        W.ln2_b[l] = ptr; ptr += C;

        W.fc_w[l] = ptr; ptr += 4LL * C * C;
        W.fc_b[l] = ptr; ptr += 4LL * C;

        W.mlp_proj_w[l] = ptr; ptr += 1LL * C * 4 * C;
        W.mlp_proj_b[l] = ptr; ptr += C;
    }

    W.ln_f_w = ptr; ptr += C;
    W.ln_f_b = ptr; ptr += C;

    W.lm_head_w = ptr;
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {

    if (argc < 3) {
        printf("ERROR_ARGS");
        return 1;
    }

    FILE* f = fopen("model.bin", "rb");
    if (!f) {
        printf("ERROR_MODEL_NOT_FOUND");
        return 1;
    }

    fread(&cfg, sizeof(int), 5, f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 5 * sizeof(int), SEEK_SET);

    model_data_buffer = (float*)malloc(file_size - 5 * sizeof(int));
    fread(model_data_buffer, 1, file_size - 5 * sizeof(int), f);
    fclose(f);

    map_weights(model_data_buffer);

    std::vector<int> input_ids;
    char* token = strtok(argv[1], ",");
    while (token) {
        input_ids.push_back(atoi(token));
        token = strtok(NULL, ",");
    }

    if (input_ids.size() >= (size_t)cfg.block_size)
        input_ids.resize(cfg.block_size - 1);

    int max_new_tokens = atoi(argv[2]);

    float temperature = (argc > 3) ? atof(argv[3]) : 0.8f;
    int top_k = (argc > 4) ? atoi(argv[4]) : 40;
    if (temperature < 0.01f) temperature = 0.01f;
    if (top_k < 1) top_k = 1;
    if (top_k > cfg.vocab_size) top_k = cfg.vocab_size;

    srand((unsigned int)time(NULL));

    const int C = cfg.n_embd;

    KVCache kv;
    kv.k_cache = (float*)calloc((long long)cfg.n_layer * cfg.block_size * C, sizeof(float));
    kv.v_cache = (float*)calloc((long long)cfg.n_layer * cfg.block_size * C, sizeof(float));

    float* x = (float*)malloc(C * sizeof(float));
    float* buf = (float*)malloc(C * sizeof(float));
    float* qkv_buf = (float*)malloc(3 * C * sizeof(float));
    
    // Allocate enough space for ALL heads to process simultaneously
    float* attn_buf = (float*)malloc(cfg.n_head * cfg.block_size * sizeof(float));
    
    float* ff_buf = (float*)malloc(4 * C * sizeof(float));
    float* logits = (float*)malloc(cfg.vocab_size * sizeof(float));

    for (int i = 0; i < (int)input_ids.size(); i++)
        forward(input_ids[i], i, &kv, x, buf, qkv_buf, attn_buf, ff_buf, logits);

    int pos = input_ids.size();

    for (int i = 0; i < max_new_tokens; i++) {

        if (pos >= cfg.block_size)
            break;

        for (int v = 0; v < cfg.vocab_size; v++)
            logits[v] /= temperature;

        std::vector<std::pair<float, int>> pairs(cfg.vocab_size);
        for (int v = 0; v < cfg.vocab_size; v++)
            pairs[v] = {logits[v], v};

        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                return a.first > b.first;
            });

        float sum = 0.0f;
        for (int j = 0; j < top_k; j++) {
            pairs[j].first = expf(pairs[j].first);
            sum += pairs[j].first;
        }
        for (int j = 0; j < top_k; j++)
            pairs[j].first /= sum;

        float r = (float)rand() / ((float)RAND_MAX + 1.0f);
        float cum = 0.0f;
        int best = pairs[0].second;
        for (int j = 0; j < top_k; j++) {
            cum += pairs[j].first;
            if (r < cum) {
                best = pairs[j].second;
                break;
            }
        }

        printf("%d ", best);

        if (best == 50256)
            break;

        forward(best, pos, &kv, x, buf, qkv_buf, attn_buf, ff_buf, logits);
        pos++;
    }

    free(model_data_buffer);
    return 0;
}