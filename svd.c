#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <lapacke.h>
#include <math.h>
#include <stdint.h>

//extern void dsyev( char* jobz, char* uplo, int* n, double* a, int* lda,double* w, double* work, int* lwork, int* info);
                
//on pourra générer des double aléatoire entre 0.0 et LIMIT_DOUBLE
#define LIMIT_DOUBLE 1.0
#define A_ 1
#define B_ 9
#define C_ 10891089
#define D_ 8910891
#define E_ 11108889
#define F_ 9089091
#define G_ 10888911
#define H_ 8909109
#define AFFICHE


#ifdef __i386
uint64_t rdtsc() {
  uint64_t x;
  __asm__ volatile ("rdtsc" : "=A" (x));
  return x;
}
#elif defined __amd64
uint64_t rdtsc() {
  uint64_t a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}
#endif

void mat_A9(double *M, int n){
	int i;
	M[0] = A_;
	M[1] = B_;
	for(i=1; i<n-1; i++){
		M[i*n+i-1] = B_;
		M[i*n+i  ] = A_;
		M[i*n+i+1] = B_;
	}
	M[n*n-2] = B_;
	M[n*n-1] = A_;
}

void mat_AMn(double *M, int n){
	int i;
	M[0] =  1.0;
	M[1] = -0.1;
	for(i=1; i<n-1; i++){
		M[i*n+i-1] =  0.1;
		M[i*n+i  ] =  1.0 + i;
		M[i*n+i+1] = -0.1;
	}
	M[n*n-2] = 0.1;
	M[n*n-1] = n;
}

double rand_double()
{
	return (double)rand()/(double)(RAND_MAX/LIMIT_DOUBLE);
}

void compute_M_Mt(int m,int n, double* M, double* MMt)
{
	int i,j,k;
	for(i=0;i<m;i++)
		for(j=0;j<m;j++)
			for(k=0;k<n;k++)
				MMt[i*m+j] += M[i*n+k] * M[j*n+k];
}

void compute_Mt_M(int m,int n, double* M, double* MtM)
{
	int i,j,k;
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			for(k=0;k<m;k++)
				MtM[i*n+j] += M[k*n+i] * M[k*n+j];
}

void transpose(int m,int n,double* mat_V,double* mat_Vt)
{
	int i,j;
	for(i=0;i<m;i++)
		for(j=0;j<n;j++)
			mat_Vt[i*n+j] = mat_V[j*m +i];
}

void affiche_mat(int m, int n,double* mat)
{
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			printf("%f ",mat[i*n+j]);
		}
		printf("\n");
	}
}

void prod_mat(int m,int n, double *mat_1,int o,int p, double* mat_2,double* mat_res)
{
	if(n==o)
	{
		int i,j,k;
		for(i=0;i<m;i++)
			for(j=0;j<p;j++)
				for(k=0;k<n;k++)
					mat_res[i*p+j] += mat_1[i*n+k] * mat_2[k*p+j];
	}
	else
	{
		printf("Porduit mat*mat impossible\n");
	}
}

int main( int argc, char** argv)
{
#ifdef AFFICHE
	printf("DECOMPOSITION EN VALEURS SINGULIERES\n");
#endif
//INITIALISATION
	int size_m,size_n;
	double *mat_U,*mat_V,*mat_Vt,*mat_Ut,*sigma,*res,*res2,*res3;
	int i,j;
	uint64_t t0, t1, t2, t3;

	if(argc < 3){
		printf("Argument manquant !!! <./bin size_m size_n>\n");
		return 1;
	}

	size_m = atoi(argv[1]);
	size_n = atoi(argv[2]);

	double *mat_M;
	mat_M = (double*)malloc(size_m*size_n*sizeof(double));

	mat_U  = (double*)calloc(size_m*size_m, sizeof(double));
	mat_V  = (double*)calloc(size_n*size_n, sizeof(double));
	mat_Vt = (double*)calloc(size_n*size_n, sizeof(double));
	mat_Ut = (double*)calloc(size_m*size_m, sizeof(double));
	sigma  = (double*)calloc(size_m*size_n, sizeof(double));
	res    = (double*)calloc(size_m*size_n, sizeof(double));
	res2   = (double*)calloc(size_m*size_n, sizeof(double));
	res3   = (double*)calloc(size_n*size_n, sizeof(double));
	srand(time(NULL));
/*
	for(i=0;i<size_m;i++)
		for(j=0;j<size_n;j++)
			mat_M[i*size_n + j] = rand_double();

	double mat_M[] = {1.0,0.0,0.0,0.0,2.0,
					  0.0,0.0,3.0,0.0,0.0,
					  0.0,0.0,0.0,0.0,0.0,
					  0.0,4.0,0.0,0.0,0.0};
*/

	mat_A9(mat_M, size_m);

#ifdef AFFICHE
	printf("Matrice M\n");
	affiche_mat(size_m,size_n,mat_M);
#endif	
	t0 = rdtsc();
//COMPUTE U
	compute_M_Mt(size_m,size_n,mat_M,mat_U);
#ifdef AFFICHE
/*	printf("Matrice U\n");
	affiche_mat(size_m,size_m,mat_U);*/
#endif
//COMPUTE V*
	compute_Mt_M(size_m,size_n,mat_M,mat_V);
	//transpose(size_n,size_n,mat_V,mat_Vt);
#ifdef AFFICHE
	/*printf("Matrice V\n");
	affiche_mat(size_n,size_n,mat_V);*/
#endif	
	
//COMPUTE U EIGEN VALUES 
	//GSL
	gsl_matrix_view m  = gsl_matrix_view_array (mat_U, size_m, size_m);
	gsl_vector *eval = gsl_vector_alloc (size_m);
	gsl_matrix *evec = gsl_matrix_alloc (size_m, size_m);
	gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc (size_m);
	gsl_eigen_symmv (&m.matrix, eval, evec, w);
	gsl_eigen_symmv_free (w);
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_DESC);
	
	for (i = 0; i < size_m; i++)
	  {
		double eval_i 
		   = gsl_vector_get (eval, i);
		gsl_vector_view evec_i 
		   = gsl_matrix_column (evec, i);
		sigma[i*size_n+i] = sqrt(eval_i);
#ifdef AFFICHE
/*
		printf("eigenvalue = %g\n", eval_i);
		printf("eigenvector = \n");
		gsl_vector_fprintf(stdout, &evec_i.vector, "%g");
*/
#endif
	}
#ifdef AFFICHE
	printf("Matrice U\n");
	affiche_mat(size_m, size_m, evec->data);
#endif
	
	prod_mat(size_m,size_m,evec->data,size_m,size_n,sigma,res);
	
	m  = gsl_matrix_view_array (mat_V, size_n, size_n);
	eval = gsl_vector_alloc (size_n);
	evec = gsl_matrix_alloc (size_n, size_n);
	w = gsl_eigen_symmv_alloc (size_n);
	gsl_eigen_symmv (&m.matrix, eval, evec, w);
	gsl_eigen_symmv_free (w);
	gsl_eigen_symmv_sort (eval, evec, GSL_EIGEN_SORT_VAL_DESC);
			
	t1 = rdtsc();
#ifdef AFFICHE
	printf("Matrice sigma\n");
	affiche_mat(size_m, size_n, sigma);
#endif
	transpose(size_n, size_n, evec->data, mat_Vt);
#ifdef AFFICHE
	printf("Matrice V*\n");
	affiche_mat(size_n, size_n, mat_Vt);
#endif
	prod_mat(size_m, size_n, res, size_n, size_n, mat_Vt, res2);
#ifdef AFFICHE
	printf("Matrice recalculée\n");
	affiche_mat(size_m,size_n,res2);
#endif
	gsl_vector_free (eval);
	gsl_matrix_free (evec);
	
	//BLAS
	/*int m = size_m;
	int n = size_n;
	int info,lwork=-1;
	double wkopt;
	double* vector = (double*)malloc(m*sizeof(double));
	double* vector2 = (double*)malloc(size_n*sizeof(double));
	LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V','U',m,mat_U,m,vector);
	LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V','U',size_n,mat_V,size_n,vector2);
	printf("Matrice U\n");
	affiche_mat(m,m,mat_U);
	printf("Matrice V\n");
	affiche_mat(n,n,mat_V);*/
	
	return 0;
}
