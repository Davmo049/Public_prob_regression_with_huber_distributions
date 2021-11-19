#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "welzl.h"

static void shuffle(double* points, int N, int d);
static double L2(double* p2, int d);
static double L2_diff(double* p1, double* p2, int d);
static void sub_mat_vec(double* dst, double *src, double* v, int n, int d);
static void vec_add(double* dst, double* a, double* b, int n);
static void inplace_mul(double* v, double factor, int n);
static void matmul(double* dst, double* A, double* Bt, int M, int K, int N);
static void welzl_trivial(struct welzl_circle* ret, double* points, int N, int d);
static void inplace_transpose(double* A, int n);
static void printmat(double* A, int r, int c);

extern void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv,
                double* b, int* ldb, int* info );

void welzl_plain(double* center, double* radius, double* points, int N, int d)
{
    struct welzl_circle ret;
    ret.p=center;
    welzl_iterative(&ret, points, N, d);
    radius[0] = ret.radius;
}

void welzl_iterative(struct welzl_circle* ret, double* points, int N, int d)
{
    shuffle(points, N, d);
    int c_idx = N;
    char* buff = (char*) malloc(sizeof(int)*(d+1)+sizeof(double)*(d+1)*d);
    int* included_indices = (int*) buff;
    double* included_points = (double*) (included_indices+d+1);
    int inc_len = 0;
    ret->radius = 0.0;
    memset(ret->p, 0, sizeof(double)*d);
    if(N == 0)
    {
        free(buff);
        return;
    }
    double eps = 0.000001;
    while(1)
    {
        if(c_idx == N)
        {
            welzl_trivial(ret, included_points, inc_len, d);
            c_idx -= 1;
        }
        else if(L2_diff(ret->p, points+(c_idx*d), d) < ret->radius+eps)
        {
            if(c_idx == 0)
            {
                free(buff);
                return;
            }
            else
            {
                c_idx -= 1;
            }
        }
        else
        {
            for(int i=0; i < inc_len; ++i)
            {
                if(included_indices[i] > c_idx)
                {
                    inc_len = i;
                    break;
                }
            }
            if(inc_len == 0 || included_indices[inc_len-1] < c_idx)
            {
                included_indices[inc_len] = c_idx;
                memcpy(included_points+(d*inc_len), points+(d*c_idx), sizeof(double)*d);
                inc_len += 1;
                if(inc_len == d+1)
                {
                    welzl_trivial(ret, included_points, inc_len, d);
                    c_idx -= 1;
                }
                else
                {
                    c_idx = N;
                }
            }
            else
            {
                while(included_indices[inc_len-1] == c_idx)
                {
                    inc_len -= 1;
                    c_idx -= 1;
                }
                included_indices[inc_len] = c_idx;
                memcpy(included_points+(d*inc_len), points+(d*c_idx), sizeof(double)*d);
                inc_len += 1;
                c_idx = N;
            }
        }
    }
}

static double L2(double* p, int d)
{
    double r = 0.0;
    for(int i=0; i<d; ++i)
    {
        r += p[i]*p[i];
    }
    return r;
}

static double L2_diff(double* p1, double* p2, int d)
{
    double r = 0.0;
    for(int i=0; i<d; ++i)
    {
        double diff = p1[i]-p2[i];
        r += diff*diff;
    }
    return r;
}

static void sub_mat_vec(double* dst, double *src, double* v, int n, int d)
{
    // fill out nxd vector with dst = src[i, j] - v[j]
    for(int i=0; i<n; ++i){
        for(int di=0; di<d;++di) {
            dst[i*d+di] = src[i*d+di]-v[di];
        }
    }
}

static void vec_add(double* dst, double* a, double* b, int n)
{
    for(int i=0; i<n;++i)
    {
        dst[i] = a[i]+b[i];
    }
}

static void inplace_mul(double* v, double factor, int n)
{
    for(int i=0; i<n;++i)
    {
        v[i] *= factor;
    }
}

static void matmul(double* dst, double* A, double* Bt, int M, int K, int N)
//A is Mxk
//Bt is Nxk
{
    for(int r=0; r<M;++r)
    {
        for(int c=0; c<N;++c)
        {
            double v = 0.0;
            for(int k=0; k<K;++k)
            {
                v += A[r*K+k]*Bt[c*K+k];
            }
            dst[r*N+c] = v;
        }
    }
}

static void welzl_trivial(struct welzl_circle* ret, double* points, int N, int d)
{
    if(N == 0)
    {
        ret->radius=0;
        return;
    }
    else if(N == 1)
    {
        ret->radius=0;
        memcpy(ret->p, points, sizeof(double)*d);
        return;
    }
    else if(N == 2)
    {
        vec_add(ret->p, points, points+d, d);
        inplace_mul(ret->p, 0.5, d);
    }
    else if(N-1 == d)
    {
        char* buff = (char*) malloc(sizeof(double)*(d*d+d)+sizeof(int)*d);
        double* A = (double*) buff;
        double* b = A+d*d;
        int* ipiv = (int*) (b+d);
        sub_mat_vec(A, points, points+d*d, d, d);
        inplace_transpose(A,d);

        double last_norm = L2(points+(N-1)*(N-1), d);
        for(int i=0; i<N-1; ++i)
        {
            b[i] = (L2(points+d*i, d)-last_norm)/2;
        }

        //solve A x=b solution stored in ret.p
        int info;
        int one = 1;
        dgesv_(&d, &one, A, &d, ipiv, b, &d, &info);
        if(info != 0)
        {
            // we will reject this
            memset(ret->p, 0, sizeof(double)*d);
            ret->radius = -1;
            free(buff);
            return;
        }
        memcpy(ret->p, b, sizeof(double)*d);
        free(buff);
    }
    else
    {
        double* buff = (double*) malloc(sizeof(double)*((N-1)*d+(N-1)*(N-1)+(N-1))+sizeof(int)*(N-1));
        double* points_diff = buff;
        double* A = points_diff + (N-1)*d;
        double* b = A + (N-1)*(N-1);
        int* ipiv = (int*) (b+(N-1));

        sub_mat_vec(points_diff, points, points+(N-1)*d, N-1, d);

        matmul(A, points_diff, points_diff, N-1, d, N-1);

        double last_norm = L2(points+(N-1)*d, d);

        for(int i=0; i<N-1; ++i){
            b[i] = L2_diff(points+d*i, points+(N-1)*d, d)/2;
        }
        // solve A x = b
        // Then ret.p = matmul(ret.p, x, points_diff);
        // vec_add(ret.p, ret.p, points+d*(N-1), d);
        int info;
        int nAct = N-1;
        int one = 1;
        dgesv_(&nAct, &one, A, &nAct, ipiv, b, &nAct, &info);
        if(info != 0)
        {
            // we will reject this
            memset(ret->p, 0, sizeof(double)*d);
            ret->radius = -1;
            free(buff);
            return;
        }
        //matmul(ret->p, b, points_diff, 1, N-1, d);
        for(int i=0; i<d; ++i)
        {
            double v = 0.0;
            for(int j=0; j<N-1; ++j)
            {
                v += b[j]*points_diff[j*d+i];
            }
            ret->p[i] = v;
        }
        vec_add(ret->p, ret->p, points+((N-1)*d), d);
        free(buff);
    }
    double max = 0.0;
    for(int i=0; i < N; ++i)
    {
        double cand = 0.0;
        double* cur_p = points + i*d;
        for(int di=0; di<d; ++di)
        {
            double diff = cur_p[di]-(ret->p[di]);
            cand += diff*diff;
        }
        if(cand > max)
        {
            max = cand;
        }
    }
    ret->radius = max;
}

static void inplace_transpose(double* A, int n)
{
    double tmp;
    for(int i=0; i < n-1;++i)
    {
        for(int j=i+1; j < n;++j)
        {
            tmp = A[n*i+j];
            A[n*i+j] = A[n*j+i];
            A[n*j+i] = tmp;
        }
    }
}

static void shuffle(double* buf, int N, int d)
{
    double tmp;
    for(int i=0; i < N; ++i)
    {
        int swap_idx = rand()%(N-i)+i; // biased uniform, probably good enough
        if(swap_idx != i)
        {
            for(int j=0; j < d; ++j)
            {
                // swap
                tmp = buf[i*d+j];
                buf[i*d+j] = buf[swap_idx*d+j];
                buf[swap_idx*d+j] = tmp;
            }
        }
    }
}


static void printmat(double* A, int r, int c)
{
    for(int i = 0; i < r; ++i)
    {
        for(int j = 0; j < c; ++j)
        {
            if(j != c-1)
            {
                printf("%f, ",A[i*c+j]);
            }
            else
            {
                printf("%f\n",A[i*c+j]);
            }
        }
    }
}

