
#ifndef HEAD_H
#define HEAD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mmio.h"
#include <omp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <sys/stat.h>
#include <sys/types.h>



int mkdir(const char *pathname, mode_t mode);


//int num_condit=5;// change this value to the corresponding condition count


//#define num_condit 3

int fused_type=1;

int num_lambda1=10;
int num_lambda2=10;
int rounds=50;
int num_lambda1_ss=10;
int num_lambda2_ss=10;
int folds=5;

//double lambda1_max,lambda1_min;



const int MAX_ITER  = 1000;//8000
const int MAX_ITER1 = 10000;
const double RELTOL = 1e-4;//1e-3;
const double ABSTOL = 1e-5;//1e-4;
//const double RELTOL = 1e-5;
//const double ABSTOL = 1e-6;



void Build_D_matrix(gsl_matrix  *D,int p,int num_condit,int fused_type,double lambda1,double lambda2);
void rand_num(gsl_vector *fold_id);
double PowerMethod(gsl_matrix *A,int maxiter,double tol,int n_threads);
double objective(gsl_matrix *A, gsl_vector *b, double lambda1, double lambda2,gsl_vector *x,gsl_vector *z,int num_condit0);
void FISTA(gsl_matrix *x,gsl_vector *y, gsl_matrix *X_lassotX_lasso, gsl_vector *X_lassotY, double lambda,gsl_vector *beta,double Lip_lasso);
void AIMNet_Solver(int N_threads,gsl_matrix *X, gsl_vector *Y,gsl_vector *beta, gsl_vector *z,gsl_matrix *D,gsl_matrix *Dt,gsl_matrix *DtD,
                       double rho,double lambda1,double lambda2,int MAX_ITER,double Lip);

void ProAdmmLasso(gsl_matrix *X,gsl_matrix *XtX,gsl_vector *XtY,gsl_vector *Y,gsl_vector *beta_lasso,
                 double lambda1,double rho,double Lip_lasso,int MAX_ITER,int N_threads);

int simu_2condits(char *argv[]);
int simu_3condits(char *argv[]);

int simu_4condits(char *argv[]);
int simu_5condits(char *argv[]);





void Build_D_matrix(gsl_matrix  *D,int p,int num_condit,int fused_type,double lambda1,double lambda2){
// x is p*1 vector, there are num_condit conditions
// fused_type means which kind of fused term
// 1 means successive difference(1D fused term)
// 2 means the permutations of any two conditions(2D fused term )

/*we build the D matrix to combined the lasso and fused lasso term*/
/*fused_type==1*/
/*
*       lambda1/lambda2                 0                        0
*       0                        lambda1/lambda2                 0
*       0                               0                 lambda1/lambda2
*       1                              -1                        0
*       0                               1                       -1
*
*/
/*fused_type==2*/
/*
*       lambda1/lambda2                 0                        0
*       0                        lambda1/lambda2                 0
*       0                               0                 lambda1/lambda2
*       1                              -1                        0
*       1                               0                       -1
*       0                               1                       -1
*
*/

/*int m,n;
if(type_fused==1){
    m=num_condit*p+p*(num_condit-1);
}
if(type_fused==2){
  m=comb_num(num_condit,2);
}
n=num_condit*p;*/


int i,j,t;

if(fused_type==0){// fused_type==0 means lasso problem
   for(i=0;i<(num_condit*p);i++){
      gsl_matrix_set(D,i,i,1);
      //gsl_matrix_set(D,i,i,gsl_vector_get(x_sd_recip,i));
    }
    printf("*****************\n");
   printf("Build D for lasso problem!\n");

}else{

    if(fused_type==1){// 1_D fused type

        for(t=1;t<num_condit;t++){
            for(i=((t-1)*p);i<(p*t);i++){
                for(j=(t-1)*p;j<t*p;j++){
                    if((i-(t-1)*p)==(j-(t-1)*p)){
                        //gsl_matrix_set(D,i,j,r);
                       //gsl_matrix_set(D,i,j+p,-r);
                       //gsl_matrix_set(D,i,j,gsl_vector_get(x_sd_recip,j));
                       //gsl_matrix_set(D,i,j+p,-gsl_vector_get(x_sd_recip,j+p));

                       gsl_matrix_set(D,i,j,1);
                       gsl_matrix_set(D,i,j+p,-1);
                    }
                }
            }
        }

        printf("*****************\n");

    //    for(t=1;t<(num_condit);t++){
    //            for(i=((t-1)*p);i<(p*t);i++){
    //
    //                for(j=(t-1)*p;j<t*p;j++){
    //                        if((i-(t-1)*p)==(j-(t-1)*p)){gsl_matrix_set(D,i,j,1);
    //                                             gsl_matrix_set(D,i,j+p,-1);
    //                    }
    //
    //                }
    //
    //            }
    //    }



    }

    if(fused_type==2){ // 2_D fused type

        int t1,t2;
        int t3=1;
        for(t1=1;t1<num_condit;t1++){
            for(t2=t1+1;t2<=num_condit;t2++){
                 for(i=((t3-1)*p);i<(p*t3);i++){
                    for(j=(t1-1)*p;j<t1*p;j++){
                    if((i-(t3-1)*p)==(j-(t1-1)*p)){
//                            gsl_matrix_set(D,i,j,r);
//                            gsl_matrix_set(D,i,j+(t2-t1)*p,-r);
                            //gsl_matrix_set(D,i,j,gsl_vector_get(x_sd_recip,j));
                            //gsl_matrix_set(D,i,j+(t2-t1)*p,-gsl_vector_get(x_sd_recip,j+(t2-t1)*p));

                            gsl_matrix_set(D,i,j,1);
                            gsl_matrix_set(D,i,j+(t2-t1)*p,-1);
                        }
                 }

            }
             t3++;
            }
        }

    }



}


 // print D to check
//    for(i=0;i<D->size1;i++)
//    for(j=0;j<D->size2;j++){
//        printf("%lf ",gsl_matrix_get(D,i,j));
//        if(j==(D->size2-1))printf("\n");
//    }
//    printf("*************************\n");

}


/**PowerMethod to calculate the maximum singular value**/
double PowerMethod(gsl_matrix *A,int maxiter,double tol,int n_threads){
/// compute the maximum eigenvalue of matrix A
/* the default values of maxiter and tol can choose 1000 and 5e-3*/
/*A is n*n square matrix*/
int n=A->size1;
int iter=0,i=0,j=0;
double lambda=0;
double temp=0;
double a1=0,a2=0;
double aa[A->size1][A->size2];
double bb[A->size2];
double c[A->size1];




gsl_vector *v_old=gsl_vector_calloc(n);
gsl_vector *v_new=gsl_vector_calloc(n);
gsl_vector *z_new=gsl_vector_calloc(n);
gsl_vector *w=gsl_vector_calloc(n);
gsl_vector *Abyv_new=gsl_vector_calloc(n);


/*initialize aa=A^T, c=0, v_old=1,v_new=1 in parallel*/
 //   #pragma omp parallel num_threads(n_threads) private(i,j) shared(aa,bb,A,c,v_old,v_new)
    {
   //    #pragma omp for schedule(static,n/n_threads)
       for(i=0;i<n;i++){
        for(j=0;j<n;j++){
            aa[i][j]=gsl_matrix_get(A,i,j);
        }
        c[i]=0;
        bb[i]=0;
        gsl_vector_set(v_old,i,i+0.11);
        gsl_vector_set(v_new,i,i+0.12);
        //printf("i=%d,ThreadId=%d\n",i,omp_get_thread_num());
    }

    }

while(iter<=maxiter){

 //  #pragma omp parallel num_threads(n_threads) private(i) shared(bb,v_old)
    {
     //   #pragma omp for schedule(static,n/n_threads)
        for(i=0;i<n;i++){
            bb[i]=gsl_vector_get(v_old,i);
        }

    }
  // #pragma omp parallel for schedule(static,n/n_threads) private(i,j,temp) shared(z_new,w,aa,bb,c)
        for(i=0;i<n;i++){
            temp=0;
            for(j=0;j<n;j++){
            temp+=aa[i][j]*bb[j];
            }
        c[i]=temp;
        gsl_vector_set(z_new,i,c[i]);
        gsl_vector_set(w,i,c[i]);

        }


    //gsl_blas_dgemv(CblasNoTrans, 1, A, v_old, 0, z_new);//z_new=A*v_old
    //gsl_vector_memcpy(w, z_new);
    if(gsl_blas_dnrm2(z_new)==0)
        {printf("Error: The PowerMethod has a problem:gsl_blas_dnrm2(z_new) is zero!\n");}
    gsl_vector_scale(w,1/gsl_blas_dnrm2(z_new));
    gsl_vector_memcpy(v_new, w);
    gsl_vector_sub(w, v_old);

    temp=gsl_blas_dnrm2(w);//dnrm2(w) is L2_norm=sqrt(sum(w^2))
    if(temp<=tol)
    {
    break;
    }
    gsl_vector_memcpy(v_old, v_new);
    iter++;
}

gsl_blas_dgemv(CblasNoTrans, 1, A, v_new, 0, Abyv_new);
gsl_blas_ddot(Abyv_new,v_new,&a1);
gsl_blas_ddot(v_new,v_new,&a2);
lambda=a1/a2;

gsl_vector_free(v_old);
gsl_vector_free(v_new);
gsl_vector_free(z_new);
gsl_vector_free(w);
gsl_vector_free(Abyv_new);

return lambda;
}

/*generate random number*/
void rand_num(gsl_vector *fold_id){
    int n=fold_id->size,i,w,t;
    int a[n];
    for(i=0;i<n;++i){
        a[i]=i;
    }

for(i=(n-1);i>=1;--i){
   w=rand()%i;
   t=a[w];
   a[w]=a[i];
   a[i]=t;
}
for(i=0;i<n;i++){
   gsl_vector_set(fold_id,i,a[i]);
}

//free(&n);
//free(&a);
}


/**Calculate the value of objective function**/
double objective(gsl_matrix *A, gsl_vector *b, double lambda1, double lambda2,gsl_vector *x,gsl_vector *z,int num_condit0) {

	int m_A=A->size1;
	int n_A=A->size2;
	int i,j;
    double obj = 0;
	double Abeta_b_nrm2,temp1;
	gsl_vector *Abeta_b = gsl_vector_calloc(m_A);
	gsl_vector *beta= gsl_vector_calloc(n_A);



	if(num_condit0==1){// only one condition, Lasso problem
        /// objective function: ||Ax-b||^2+lambda1*||b||_1
        for(int i=0;i<(n_A);i++){
        gsl_vector_set(beta,i,gsl_vector_get(x,i));//the up part of z is lasso term, and the down part is fused term
	    }

	//gsl_blas_dgemv(CblasNoTrans, 1, A, beta, 0, Abeta_b);
	//gsl_vector_sub(Abeta_b, b);
	//gsl_blas_ddot(Abeta_b, Abeta_b, &Abeta_b_nrm2);
	// #pragma omp parallel for private(i,j,temp1) shared(Abeta_b,A,beta,b)
        for(i=0;i<m_A;i++){
            temp1=0;
            for(j=0;j<n_A;j++){
            temp1+=gsl_matrix_get(A,i,j)*gsl_vector_get(beta,j);
            }
            gsl_vector_set(Abeta_b,i,temp1-gsl_vector_get(b,i));
        }

        temp1=0;
     //   #pragma omp parallel for private(i) shared(temp1)
        for(i=0;i<m_A;i++){
     //       #pragma omp  critical
            {
                temp1+=gsl_vector_get(Abeta_b,i)*gsl_vector_get(Abeta_b,i);
            }

        }
        Abeta_b_nrm2=temp1;

        obj = Abeta_b_nrm2+lambda1*gsl_blas_dasum(beta);

	} else if(num_condit0==2){// ADMM lasso, X used the raw data divided by standard deviation under different conditions
      /// objective function: ||Ax-b||^2+lambda1*||b||_1+lambda2*||b_1-b_2||_1
	 for(int i=0;i<(n_A);i++){
        gsl_vector_set(beta,i,gsl_vector_get(x,i));//the up part of z is lasso term, and the down part is fused term
	    }
	  //  #pragma omp parallel for private(i,j,temp1) shared(Abeta_b,A,beta,b)
        for(i=0;i<m_A;i++){
            temp1=0;
            for(j=0;j<n_A;j++){
            temp1+=gsl_matrix_get(A,i,j)*gsl_vector_get(beta,j);
            }
            gsl_vector_set(Abeta_b,i,temp1-gsl_vector_get(b,i));
        }

        temp1=0;
     //   #pragma omp parallel for private(i) shared(temp1)
        for(i=0;i<m_A;i++){
      //      #pragma omp  critical
            {
                temp1+=gsl_vector_get(Abeta_b,i)*gsl_vector_get(Abeta_b,i);
            }

        }
        Abeta_b_nrm2=temp1;
        obj =  Abeta_b_nrm2+lambda1*gsl_blas_dasum(beta)+lambda2*gsl_blas_dasum(z);
	}else {
         //printf("calculate the values of objective function\n");
	     gsl_vector *fused_term=gsl_vector_calloc((z->size));
         for(i=0;i<(x->size);i++){
         gsl_vector_set(beta,i,gsl_vector_get(x,i));//the up part of z is lasso term, and the down part is fused term
	     }

         for(i=0;i<((z->size));i++){
            gsl_vector_set(fused_term,i,gsl_vector_get(z,i));
        }

	     //gsl_blas_dgemv(CblasNoTrans, 1, A, beta, 0, Abeta_b);
	//gsl_vector_sub(Abeta_b, b);
	//gsl_blas_ddot(Abeta_b, Abeta_b, &Abeta_b_nrm2);
	// #pragma omp parallel for private(i,j,temp1) shared(Abeta_b,A,beta,b)
        for(i=0;i<m_A;i++){
            temp1=0;
            for(j=0;j<n_A;j++){
            temp1+=gsl_matrix_get(A,i,j)*gsl_vector_get(beta,j);
            }
            gsl_vector_set(Abeta_b,i,temp1-gsl_vector_get(b,i));
        }

        temp1=0;
    //    #pragma omp parallel for private(i) shared(temp1)
        for(i=0;i<m_A;i++){
     //       #pragma omp  critical
            {
                temp1+=gsl_vector_get(Abeta_b,i)*gsl_vector_get(Abeta_b,i);
            }

        }
        Abeta_b_nrm2=temp1;
        obj = 0.5 * Abeta_b_nrm2+lambda1*gsl_blas_dasum(beta)+lambda2*gsl_blas_dasum(fused_term);
        //obj = 0.5 * Abeta_b_nrm2 +lambda2*gsl_blas_dasum(z);
        gsl_vector_free(fused_term);
	}

	gsl_vector_free(Abeta_b);
	gsl_vector_free(beta);
	return obj;
}




void AIMNet_Solver(int N_threads,gsl_matrix *X, gsl_vector *Y,gsl_vector *beta, gsl_vector *z,gsl_matrix *D,gsl_matrix *Dt,gsl_matrix *DtD,double rho,double lambda1,double lambda2,int MAX_ITER,double Lip){

    /// objective function is ||X*beta-Y||^2+lambda1*||beta||_1+lambda2*||beta1-beta2||_1
    /// Lipschitz constant: L=maximum_eigenvalue(2*X^TX)+maximum_eigenvalue(rho*D^TD)
    /// f(beta)=||X*beta-Y||^2+rho/2*||D*beta-z+u||^2
    /// gradient of f: g_f=2X^TX*beta-2*X^TY+rho*D^T(D*beta-z+u)
    /// beta: sign(beta_t-1/L*g_f)*max(||beta_t-1/l*g_f||_1-lambda1/L,0)
    /// z   : sign(D^beta+u)*max(||D*beta+u||_1,0)
    /// u   : u_t+beta-z
    int i=0,j=0;
    int m_D=D->size1;
    int n_D=D->size2;
    int n=n_D;
    int iter=0;
    double obj=0.0;
    double nbeta_stack  = 0;
    double nu_stack  = 0;
    double prires   = 0;//primal residuals
    double dualres  = 0;//dual residuals
    double eps_pri  = 0;
    double eps_dual = 0;
    double wk=0.0;
    double temp1=0,temp2=0,temp3=0,temp=0;
    double Temp1[m_D],Temp2[m_D],Temp3[n_D];

    //gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *DtDbeta  = gsl_vector_calloc(n_D);//D^T*D*beta
    gsl_vector *Dtu_z  = gsl_vector_calloc(n_D);
    gsl_vector *Dtu    = gsl_vector_calloc(n_D);
    gsl_vector *Dbeta     = gsl_vector_calloc(m_D);//D*beta
    gsl_vector *u      = gsl_vector_calloc(m_D);
    gsl_vector *beta_prev      = gsl_vector_calloc(n_D);

    gsl_vector *r      = gsl_vector_calloc(m_D);
    gsl_vector *zprev  = gsl_vector_calloc(m_D);
    gsl_vector *zdiff  = gsl_vector_calloc(m_D);
    gsl_vector *Dt_zdiff  = gsl_vector_calloc(n_D);
    gsl_vector *Qq     = gsl_vector_calloc(n);
    gsl_vector *XtY    = gsl_vector_calloc(n);
    gsl_matrix *XtX    = gsl_matrix_calloc(n,n);

   // double obj_t1=1.0,obj_t2=1.0,obj_t3=1.0,obj_t4=1.0,obj_t5=1.0,obj_t6=1.0;

//    double tk=1.0;
//    double tk_1=1.0;

    /* Precompute and cache factorizations */
    gsl_blas_dgemv(CblasTrans, 1, X, Y, 0, XtY); // XtY = X^T Y
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X,0.0, XtX);// XtX=X^TX
    printf("%3s %10s %10s %10s %10s %10s\n", " #", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

    /*initialization*/
    /// the loop iteration variable i is private and z,u,beta_prev are shared by default
    #pragma omp parallel for num_threads(N_threads)
    for(int i=0;i<(z->size);i++){
        gsl_vector_set(z,i,0);
       // printf("i is: %d, ThreadId=%d\n",i,omp_get_thread_num());
    }
    #pragma omp parallel for num_threads(N_threads)
    for(int i=0;i<(u->size);i++){
        gsl_vector_set(u,i,0);
    }
    #pragma omp parallel for num_threads(N_threads)
    for(int i=0;i<(beta_prev->size);i++){
        gsl_vector_set(beta_prev,i,0);
    }


    /// warm start: beta_lasso is initialized with previous values


while (iter < MAX_ITER) {

    temp1=0,temp2=0,temp3=0;

    // fast iterative proximal gradient
     wk=iter/(iter+3);
     #pragma omp parallel for private(i,temp) shared(beta_prev,beta,wk) num_threads(N_threads)
     for(i=0;i<n;i++){
     temp=gsl_vector_get(beta,i);
       temp=temp+wk*(temp-gsl_vector_get(beta_prev,i));
       gsl_vector_set(beta,i,temp);
     }

    /* beta-update: beta = sign(beta_old-(1/L)*(2*X^TX*beta_old-2*XtY+rho*D^TDbeta_old+rho*D^T(u-z)))*max(|beta_old-(1/L)*(2*X^TX*beta_old-2*XtY+rho*D^TDbeta_old+rho*D^T(u-z))|-lambda1/L,0)*/
    #pragma omp parallel for private(i,j,temp1) shared(XtX,beta,Qq) num_threads(N_threads)
    for(i=0;i<n;i++){
        temp1=0;
        for(j=0;j<n;j++){
        temp1+=gsl_matrix_get(XtX,i,j)*gsl_vector_get(beta,j);
        }
        gsl_vector_set(Qq,i,temp1);//Qq=X^TX*beta_old
   // printf("%g ",temp1);
    }

   // printf("\n");
    //printf("DtDbeta:\n");
    #pragma omp parallel for private(i,j,temp1) shared(DtD,beta,DtDbeta) num_threads(N_threads)
    for(i=0;i<n_D;i++){
        temp1=0;
        for(j=0;j<n_D;j++){
        temp1+=gsl_matrix_get(DtD,i,j)*gsl_vector_get(beta,j);
        }
        gsl_vector_set(DtDbeta,i,temp1);//D^TD*beta_old
     //   printf("%g ",temp1);
    }


   //  printf("Dtu_z:\n");
     #pragma omp parallel for private(i,j,temp1) shared(Dt,u,z,Dtu_z) num_threads(N_threads)
    for(i=0;i<n_D;i++){
        temp1=0;
        for(j=0;j<m_D;j++){
        temp1+=gsl_matrix_get(Dt,i,j)*(gsl_vector_get(u,j)-gsl_vector_get(z,j));
        }
        gsl_vector_set(Dtu_z,i,temp1);
    //    printf("%g ",temp1);

    }
    //printf("\n");
    // printf("Beta:\n");
    #pragma omp parallel for private(i,j,temp1,temp2) shared(beta) num_threads(N_threads)
    for(i=0;i<n;i++){
        temp1=2*gsl_vector_get(Qq,i)-2*gsl_vector_get(XtY,i)
            +rho*(gsl_vector_get(DtDbeta,i)+gsl_vector_get(Dtu_z,i));
        temp1=temp1/(Lip);
        temp2=gsl_vector_get(beta,i)-temp1;
        gsl_vector_set(beta_prev,i,gsl_vector_get(beta,i));
        gsl_vector_set(beta,i,temp2);//beta=beta_old-1/L(XtXbeta_old-XtY+rho*Dt(Dbeta_old+u-z))
     //   printf("%g ",temp1);
    }



    //soft-thresholding beta=sign(beta)max(|beta|-lambda1/Lip,0)
    // printf("Beta_after_soft_thresholding:\n");
     #pragma omp parallel for private(i,j,temp1) shared(beta) num_threads(N_threads)
    for(i=0;i<beta->size;i++){
        temp1=gsl_vector_get(beta,i);
     if (temp1 > (lambda1/Lip))       { gsl_vector_set(beta, i, temp1 - (lambda1/Lip)); }
        else if (temp1 < -(lambda1/Lip)) { gsl_vector_set(beta, i, temp1 + (lambda1/Lip)); }
        else              { gsl_vector_set(beta, i, 0); }
     //printf("%g ",gsl_vector_get(beta, i));
    }


     /*z-update:z=soft_threshold_{lambda2/rho}(D*beta+u)*/
    #pragma omp parallel private(i,j,temp1) shared(beta,D,Dbeta) num_threads(N_threads)
    {// it run the first for loop in parallel and then run the second for loop
     //calculate Dbeta=D*beta;
     #pragma omp for
    for(i=0;i<m_D;i++){
        temp1=0;
        for(j=0;j<n_D;j++){
        temp1+=gsl_matrix_get(D,i,j)*gsl_vector_get(beta,j);
        }
        gsl_vector_set(Dbeta,i,temp1);
    }
    }


    #pragma omp parallel for private(i,j,temp1,temp2,temp3) shared(n,z,zdiff,zprev) num_threads(N_threads)
    for(i=0;i<m_D;i++){
    temp1=gsl_vector_get(z,i);
    gsl_vector_set(zprev,i,temp1);//zprev=z
    temp2=gsl_vector_get(Dbeta,i)+gsl_vector_get(u,i);
    //gsl_vector_set(z,i,temp2);

   // printf("temp 2 is %g ",temp2);
    if (temp2 > (lambda2/rho))       { gsl_vector_set(z, i, temp2 - (lambda2/rho)); }
    else if (temp2 < -(lambda2/rho)) { gsl_vector_set(z, i, temp2 + (lambda2/rho)); }
    else              { gsl_vector_set(z, i, 0); }

    temp3=gsl_vector_get(z,i)-gsl_vector_get(zprev,i);
    gsl_vector_set(zdiff,i,temp3);//zdiff=znew-zprev
    if(iter==0)gsl_vector_set(zdiff,i,10);
    //printf("%g ",gsl_vector_get(z,i));
    }



     /* u-update: u = u + D*beta - z */
    #pragma omp parallel private(i,temp1) shared(u,z,Dbeta) num_threads(N_threads)
    {// it run the first for loop in parallel and then run the second for loop
    #pragma omp for
    for(i=0;i<m_D;i++){
      temp1=gsl_vector_get(u,i)+gsl_vector_get(Dbeta,i)-gsl_vector_get(z,i);
      gsl_vector_set(u,i,temp1);//u=u+D*beta-z
    }
    }

    /* Compute residual: r = D*beta - z */
    #pragma omp parallel for private(i,temp1) shared(r,z,Dbeta) num_threads(N_threads)
    for(i=0;i<m_D;i++){
        temp1=gsl_vector_get(Dbeta,i)-gsl_vector_get(z,i);
        gsl_vector_set(r,i,temp1);
    }
    /* Compute residual: Dtu*/

     #pragma omp parallel for private(i,j,temp1) shared(Dtu,beta,Dt) num_threads(N_threads)
    for(i=0;i<n_D;i++){
        temp1=0;
        for(j=0;j<m_D;j++){
        temp1+=gsl_matrix_get(Dt,i,j)*gsl_vector_get(u,j);
        }
        gsl_vector_set(Dtu,i,temp1);
    }
    temp1=0;temp2=0;temp3=0;
    #pragma omp parallel private(i) shared(temp1,temp2,temp3,r,Dbeta) num_threads(N_threads)
    {

          #pragma omp for
          for (i=0; i<m_D; i++)
        {
          Temp1[i]=pow(gsl_vector_get(r,i),2);
          Temp2[i]=pow(gsl_vector_get(Dbeta,i),2);
          #pragma omp  critical //critical can eliminate race conditions
          {
          temp1+=Temp1[i];
          temp2+=Temp2[i];

          }

        }

    }
        #pragma omp parallel private(i) shared(temp3) num_threads(N_threads)
    {

          #pragma omp for
          for (i=0; i<n_D; i++)
        {
          Temp3[i]=pow(gsl_vector_get(Dtu,i),2);
          #pragma omp  critical //critical can eliminate race conditions
          {
          temp3+=Temp3[i];
          }

        }

    }

    prires =sqrt(temp1);/* sqrt(sum ||r_i||_2^2) */
    nbeta_stack=sqrt(temp2);/* sqrt(sum ||Dbeta_i||_2^2) */
    nu_stack=sqrt(temp3/pow(rho, 2));/* sqrt(sum ||y_i||_2^2) */


    /* Termination checks */
    /* dual residual */
     #pragma omp parallel for private(i,j,temp1) shared(Dt,zdiff,Dt_zdiff) num_threads(N_threads)
    for(i=0;i<n_D;i++){
        temp1=0;
        for(j=0;j<m_D;j++){
        temp1+=gsl_matrix_get(Dt,i,j)*gsl_vector_get(zdiff,j);
        }
        gsl_vector_set(Dt_zdiff,i,temp1);//zdiff=D^T*(z - zprev)

    }

    dualres =  rho * gsl_blas_dnrm2(Dt_zdiff); /* ||s^k||_2= rho*||D^T*(z - zprev)||_2*/
    /* compute primal and dual feasibility tolerances */
    eps_pri  = sqrt(m_D)*ABSTOL + RELTOL * fmax(nbeta_stack, gsl_blas_dnrm2(z));
    eps_dual = sqrt(n_D)*ABSTOL + RELTOL * nu_stack;

    obj=objective(X, Y, lambda1, lambda2, beta,Dbeta,2);
    printf("%4d %10.10f %10.10f %10.15f %10.10f %10.8f\n", iter,prires, eps_pri, dualres, eps_dual, obj);

//    obj_t1=obj_t2;
//    obj_t2=obj_t3;
//    obj_t3=obj_t4;
//    obj_t4=obj_t5;
//    obj_t5=obj_t6;
//    obj_t6=obj;

     /// early stop
//     if((obj_t1<obj_t2)&&(obj_t2<obj_t3)&&(obj_t3<obj_t4)&&(obj_t4<obj_t5)&&(obj_t5<obj_t6)){
//        break;
//     }

    if ((iter!=0)&& prires <= eps_pri && dualres <= eps_dual) {
        printf("AIMNet_Solver Done! The iteration is %d\n",iter);
       // printf("eps_pri is %15.15lf, eps_dual is %15.15lf\n",eps_pri,eps_dual);





        break;
    }

    iter++;

}
   // printf("\n");
   // printf("Beta is:\n");
  for(i=0;i<(beta->size);i++){
      //if(fabs(gsl_vector_get(beta,i))<=(eps_pri/sqrt(m_D)))gsl_vector_set(beta,i,0);
      if(fabs(gsl_vector_get(beta,i))<=(2*eps_pri+4*eps_dual)/sqrt(m_D))gsl_vector_set(beta,i,0);
      temp1=gsl_vector_get(beta,i);
    //  printf("%g ",temp1);
    }

  //  printf("\n");
  //  printf("z is:\n");
    for(i=0;i<(z->size);i++){
     if(fabs(gsl_vector_get(z,i))<=(eps_dual/sqrt(n_D))) gsl_vector_set(z,i,0);
   //   printf("%g ",gsl_vector_get(z,i));
   }



//     for(int i=0;i<-n_D;i++){
//            if(fabs(gsl_vector_get(beta,i))<eps_pri){
//                gsl_vector_set(beta,i,0);
//            }
//        }
//
//        for(int i=0;i<-m_D;i++){
//            if(fabs(gsl_vector_get(z,i))<eps_dual){
//                gsl_vector_set(z,i,0);
//            }
//        }

    gsl_vector_free(DtDbeta);
    gsl_vector_free(Dtu_z);
    gsl_vector_free(Dtu);
    gsl_vector_free(Dbeta);
    gsl_vector_free(u);
    gsl_vector_free(beta_prev);

    gsl_vector_free(r);
    gsl_vector_free(zprev);
    gsl_vector_free(zdiff);
    gsl_vector_free(Dt_zdiff);
    gsl_vector_free(Qq);
    gsl_vector_free(XtY);
    gsl_matrix_free(XtX);


}




void ProAdmmLasso(gsl_matrix *X,gsl_matrix *XtX,gsl_vector *XtY,gsl_vector *Y,gsl_vector *beta_lasso,
                 double lambda1,double rho,double Lip_lasso,int MAX_ITER,int N_threads){

            int iter = 0,i;
            int n=X->size2;
            int m=n;
            double obj=0;
//            int mrow=X->size1;

            double Temp1[n],Temp2[n],Temp3[n];
            double temp1=0,temp2=0,temp3=0,temp=0;
            double nxstack  = 0;
            double nystack  = 0;
            double prires   = 0;
            double dualres  = 0;
            double eps_pri  = 0;
            double eps_dual = 0;
            gsl_vector *z   = gsl_vector_calloc(n);
            gsl_vector *beta_prev   = gsl_vector_calloc(n);
            gsl_vector *u      = gsl_vector_calloc(n);
            gsl_vector *r      = gsl_vector_calloc(n);
            gsl_vector *zprev  = gsl_vector_calloc(n);
            gsl_vector *zdiff  = gsl_vector_calloc(n);
            gsl_vector *Qq     = gsl_vector_calloc(n);
            double wk=0.0;

        //  printf("Lip_lasso is %lf\n",Lip_lasso);
        //  printf("%3s %10s %10s %10s %10s %10s\n", " #", "r norm", "eps_pri", "s norm", "eps_dual", "objective");

         for(i=0;i<(u->size);i++){
            gsl_vector_set(u,i,0);
         }
    // #pragma omp parallel for num_threads(N_threads)
      for(int i=0;i<(beta_prev->size);i++){
        gsl_vector_set(beta_prev,i,0);
      }

     //  #pragma omp parallel for num_threads(N_threads)
      for(int i=0;i<(beta_lasso->size);i++){
        gsl_vector_set(beta_lasso,i,0);
      }



while (iter < MAX_ITER) {

        temp1=0,temp2=0,temp3=0;

         // fast iterative proximal gradient
     wk=iter/(iter+3);
    // #pragma omp parallel for private(i,temp) shared(beta_prev,beta_lasso,wk) num_threads(N_threads)
     for(i=0;i<n;i++){
     temp=gsl_vector_get(beta_lasso,i);
       temp=temp+wk*(temp-gsl_vector_get(beta_prev,i));
       gsl_vector_set(beta_lasso,i,temp);
     }


		/* beta-update: beta_lasso = beta_lasso_old-(1/L)*(2*X^TX*beta_lasso_old-2*XtY+rho*(beta_lasso_old+u-z)) */
	//	#pragma omp parallel for private(i,j,temp1) shared(XtX,beta_lasso,Qq)
//        for(i=0;i<n;i++){
//            temp1=0;
//            for(j=0;j<n;j++){
//            temp1+=gsl_matrix_get(XtX,i,j)*gsl_vector_get(beta_lasso,j);
//            }
//        gsl_vector_set(Qq,i,temp1);//Qq=X^TX*beta_lasso_old
//        }

        gsl_blas_dgemv(CblasNoTrans, 1, XtX, beta_lasso, 0, Qq);



    //    #pragma omp parallel for private(i,j,temp1,temp2) shared(z,u,beta_lasso,beta_prev)
        for(i=0;i<n;i++){
            temp1=gsl_vector_get(Qq,i)*2-2*gsl_vector_get(XtY,i)+rho*(gsl_vector_get(beta_lasso,i)+gsl_vector_get(u,i)-gsl_vector_get(z,i));
            temp1=temp1/Lip_lasso;
            temp2=gsl_vector_get(beta_lasso,i)-temp1;
            gsl_vector_set(beta_prev,i,gsl_vector_get(beta_lasso,i));
            gsl_vector_set(beta_lasso,i,temp2);
            //printf("the temp1 is:%15.15lf\n",rho*(gsl_vector_get(u,i)));
        }

//      printf("the beta is:\n");
//      for(int i=0;i<(beta_lasso->size);i++){
//        printf("%g ",gsl_vector_get(beta_lasso,i));
//      }




		 /*z-update:z=soft_threshold_{lambda/rho}(beta_lasso+u)*/
    //    #pragma omp parallel for private(i,j,temp1,temp2,temp3) shared(z,beta_lasso,u)
        for(i=0;i<n;i++){
        temp1=gsl_vector_get(z,i);
        gsl_vector_set(zprev,i,temp1);//zprev=z
        temp2=gsl_vector_get(beta_lasso,i)+gsl_vector_get(u,i);
        if (temp2 > lambda1)       { gsl_vector_set(z, i, temp2 - lambda1); }
		else if (temp2 < -lambda1) { gsl_vector_set(z, i, temp2 + lambda1); }
		else              { gsl_vector_set(z, i, 0); }


        //zdiff store the difference between z and zprev
        //zdiff will be used to calculate the dual residuals
        temp3=gsl_vector_get(z,i)-gsl_vector_get(zprev,i);
        gsl_vector_set(zdiff,i,temp3);
        }


         /* u-update: u = u + beta_lasso-z */
    //    #pragma omp parallel num_threads(N_threads) private(i,temp1) shared(u,z,beta_lasso)
        {// it run the first for loop in parallel and then run the second for loop

   //     #pragma omp for
        for(i=0;i<n;i++){
          temp1=gsl_vector_get(u,i)+gsl_vector_get(beta_lasso,i)-gsl_vector_get(z,i);
          gsl_vector_set(u,i,temp1);//u=u+beta_lasso-z
          //printf("the temp1 is:%15.15lf\n",pow(gsl_vector_get(u,i),2));
        }
        }


        /* Compute residual: r = beta_lasso - z */
    //    #pragma omp parallel for private(i,temp1) shared(r)
        for(i=0;i<n;i++){
            temp1=gsl_vector_get(beta_lasso,i)-gsl_vector_get(z,i);
            gsl_vector_set(r,i,temp1);
        }

        temp1=0;temp2=0;temp3=0;
   //     #pragma omp parallel private(i) shared(temp1,temp2,temp3)
        {

    //          #pragma omp for
              for (i=0; i<n; i++)
            {
              Temp1[i]=pow(gsl_vector_get(r,i),2);
              Temp2[i]=pow(gsl_vector_get(beta_lasso,i),2);
              Temp3[i]=pow(gsl_vector_get(u,i),2);
              //printf("Temp3 is %lf ",Temp3[i]);
     //         #pragma omp  critical //critical can eliminate race conditions
              {
              temp1+=Temp1[i];
              temp2+=Temp2[i];
              temp3+=Temp3[i];
              }

            }

        }

        prires =sqrt(temp1);/* sqrt(sum ||r_i||_2^2) */
        nxstack=sqrt(temp2);/* sqrt(sum ||x_i||_2^2) */
        nystack=sqrt(temp3/pow(rho, 2));/* sqrt(sum ||y_i||_2^2) */

		/* Termination checks */
		/* dual residual */
		dualres = rho * gsl_blas_dnrm2(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */

		/* compute primal and dual feasibility tolerances */
		eps_pri  = sqrt(n)*ABSTOL + RELTOL * fmax(nxstack, gsl_blas_dnrm2(z));
		eps_dual = sqrt(m)*ABSTOL + RELTOL * nystack;
      //  printf("eps_dual is:%15.15lf\n",eps_dual);
       // printf("%4d %10.6f %10.8f %10.6f %10.8f %10.6f\n",iter,prires, eps_pri, dualres, eps_dual,objective(X, Y, lambda1, 0,beta_lasso,0,1));


		if (prires <= eps_pri && dualres <= eps_dual) {
			break;
		}

		iter++;
    }
      //  printf("\n");
      //  printf("The objective value is %10.6f\n",objective(X, Y, lambda1, 0,beta_lasso,0,1));

     for(int i=0;i<(beta_lasso->size);i++){
        gsl_vector_set(beta_lasso,i,gsl_vector_get(z,i));
      }

      obj=objective(X, Y,lambda1, 0,beta_lasso,z,1);
      printf("obj is:%15.15lf\n",obj);

        gsl_vector_free(z);
        gsl_vector_free(beta_prev);
        gsl_vector_free(u);
        gsl_vector_free(r);
        gsl_vector_free(zprev);
        gsl_vector_free(zdiff);
        gsl_vector_free(Qq);


}




void FISTA(gsl_matrix *x,gsl_vector *y, gsl_matrix *X_lassotX_lasso, gsl_vector *X_lassotY, double lambda,gsl_vector *beta,double Lip_lasso){

//y:n*1 vector, x:n*p matrix,lambda>0
//solve the problem min ||y-xb||^2+lambda*|b|_1
int n=x->size1;
int p=x->size2;
//printf("n is %d\n",n);
//printf("p is %d\n",p);
double tol=1e-4;
int MAX_iter=1000;

gsl_matrix *xtx=gsl_matrix_alloc(p,p);
gsl_vector *xty=gsl_vector_alloc(p);
gsl_vector *x_b=gsl_vector_alloc(n);
gsl_vector *x_btemp=gsl_vector_alloc(n);
gsl_vector *b=gsl_vector_alloc(p);
gsl_vector *b_1=gsl_vector_alloc(p);
gsl_vector *btemp=gsl_vector_alloc(p);
gsl_vector *b_diff=gsl_vector_alloc(p);




gsl_vector *xtx_b=gsl_vector_alloc(p);
gsl_vector *xtx_btemp=gsl_vector_alloc(p);
double temp=0,fobj=0,fobj_1=0;
gsl_vector *temp1=gsl_vector_alloc(n);
//gsl_vector *temp2=gsl_vector_alloc(n);
//gsl_vector *temp3=gsl_vector_alloc(n);
//gsl_vector *temp4=gsl_vector_alloc(p);

gsl_matrix_memcpy(xtx,X_lassotX_lasso);
gsl_vector_memcpy(xty,X_lassotY);
//gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, x,0.0, xtx);
//gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, x, y,0.0, xty);//xty=x'*y;

double s=1/Lip_lasso;//s=1/(2*trace(xtx));
double lambdas=lambda*s;

gsl_vector_memcpy(b,beta);
gsl_vector_memcpy(b_1,b);

int iter=0;
double error1=1, error2=1;
int fstop=0;


while(((fstop==0)||(error2>tol)) &&(iter<MAX_iter)){

    //btemp = b+(b-b_1)*iter/(iter+3);
    for(int i=0;i<p;i++){
       temp=gsl_vector_get(b,i)+(gsl_vector_get(b,i)-gsl_vector_get(b_1,i))*iter/(iter+3);
       gsl_vector_set(btemp,i, temp);
    }
    gsl_vector_memcpy(b_1,b);//b_1=b;


    //v=btemp-s*(xtx*btemp-xty);
    gsl_blas_dgemv(CblasNoTrans, 1, xtx, btemp,0, xtx_btemp);
    gsl_blas_dgemv(CblasNoTrans, 1, x, btemp,0, x_btemp);


//    while(1){

        for(int i=0;i<p;i++){
        temp=gsl_vector_get(btemp,i)-2*s*(gsl_vector_get(xtx_btemp,i)-gsl_vector_get(xty,i));
        //b=max(lambdas,v)+min(-lambdas,v);
        if(temp>lambdas){gsl_vector_set(b, i, temp - lambdas);}
        else if(temp<-lambdas){gsl_vector_set(b, i, temp + lambdas);}
        else                 { gsl_vector_set(b, i, 0); }
       }
       gsl_blas_dgemv(CblasNoTrans,1, x, b,0, x_b);
       gsl_blas_dgemv(CblasNoTrans, 1, xtx, b,0, xtx_b);
//       gsl_vector_memcpy(temp4,xtx_b);
//       gsl_vector_sub(temp4,xty);
//
//       gsl_vector_memcpy(b_diff,b);
//       gsl_vector_sub(b_diff,btemp);//b_diff=b-btemp
//
//       gsl_blas_ddot(temp4,b_diff,&T);
//
//       gsl_vector_memcpy(temp2,y);
//       gsl_vector_sub(temp2,x_btemp);
//
//       gsl_vector_memcpy(temp3,y);
//       gsl_vector_sub(temp3,x_b);
//
//       if(gsl_blas_dnrm2(temp2)<gsl_blas_dnrm2(temp3)+T+gsl_blas_dnrm2(b_diff)) {
//        break;
//       }
//
//
//      lambdas=0.9*lambdas;
//       gsl_vector_memcpy(temp1,y);
//        gsl_vector_sub(temp1,x_b);
//        //if(T>100) break;
//
//    }


//        gsl_vector_memcpy(temp1,y);
//        gsl_vector_sub(temp1,x_b);
//        fobj=gsl_blas_dnrm2(temp1)+lambda*gsl_blas_dasum(b);
        //printf("fobj is:%lf\n",fobj);



    if(fstop==0){
        //temp1=y-x*b;
        gsl_vector_memcpy(temp1,y);
        gsl_vector_sub(temp1,x_b);

        fobj=pow(gsl_blas_dnrm2(temp1),2)+lambda*gsl_blas_dasum(b);
        error1=fabs(fobj-fobj_1)/(1+fobj_1);
        //printf("fobj is:%lf\n",fobj);
        if(error1<tol){
          fstop=1;
          printf("fobj is:%15.15lf\n",fobj);
        }
        fobj_1=fobj;

    }



    //error2=sqrt(temp'*temp)/(1+sqrt(b_1'*b_1));

    iter=iter+1;




}
gsl_vector_memcpy(beta,b);




  gsl_matrix_free(xtx);
  gsl_vector_free(xty);
  gsl_vector_free(x_b);
  gsl_vector_free(x_btemp);
  gsl_vector_free(b);
  gsl_vector_free(b_1);
  gsl_vector_free(btemp);
  gsl_vector_free(b_diff);





}







/*
Solve the differential gene regulatory networks inference problem under two conditions.
Set the number of conditions in head.h file: num_condit=2.
The program uses OpenMP for parallel implementation and the GNU Scientific Library(GSL) for math.

Run the program in parallel:
    Use "export OMP_NUM_THREADS=num_threads" to tell the program the number of threads.
    For example:
        In the command line:
            export OMP_NUM_THREADS=16
        In the code:
            N_threads=omp_get_max_threads()-1;// which means the number of threads used in parallel is 15 (the total threads number-1).

Compile the code in local:
    gcc -Wall -std=c99 -I/GSL_path/include -c mmio.c -o mmio.o
    gcc -fopenmp -Wall -std=c99 -g -I/GSL_path/include -c simu_2condits.c -o simu_2condits.o
    gcc -fopenmp -Wall -std=c99 -g -L/GSL_path/lib  simu_2condits.o mmio.o -o simu_2condits -lgsl -lgslcblas -lm

    For example:
   	gcc -Wall -std=c99 -I/usr/local/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/usr/local/include -c simu_2condits.c -o simu_2condits.o
	gcc -fopenmp -Wall -std=c99 -g -L/usr/local/lib  simu_2condits.o mmio.o -o simu_2condits -lgsl -lgslcblas -lm

Run program:
    // export the threads used in parallel
    // demo_data: the example data, 2 is the number of tissues
    export OMP_NUM_THREADS=5
    ./simu_2condits demo_data 2
*/


int simu_2condits(char *argv[])
{
     int num_condit;
     long conv=strtol(argv[2],NULL,10); // the second parameter is the number of conditions
     num_condit=conv;
     //num_condit=(int)argv[2];


    FILE *f;
    char s[40];
    double entry,s_entry,Lip_temp;
    //double st_up,et_up,st_down,et_down;
    double Lip_data,Lip_DtD;
    double Lip,Lip_lasso,temp_i,temp;

    int i,j,k,m,m0,p,ncol_X,m_D,n_D;
    gsl_vector  *Lip_up_temp=gsl_vector_calloc(rounds);
    gsl_vector  *Lip_down_temp=gsl_vector_calloc(rounds);
    int temp_k=0;
    int ncol_y,nrow_y;
    int row,col;
    int s_m0,s_n0;
    int i_lambda1,j_lambda2;
    //int N_threads=1;
    int N_threads=omp_get_max_threads()-1;



    //double lambda1_ss,lambda2_ss;
    double lambda1_ss_max,lambda1_ss_min,lambda2_ss_max,lambda2_ss_min;
    double del_log_lambda1_ss,del_log_lambda2_ss;
    gsl_vector *Lambda1_ss=gsl_vector_calloc(num_lambda1_ss);
    gsl_matrix *Lambda2_ss=gsl_matrix_calloc(num_lambda1_ss,num_lambda2_ss);
    gsl_vector *M= gsl_vector_calloc(num_condit);//M stores the number of samples under each condition
    // M[0] is the first condition

    mkdir("solution_SS",0700);


    /**Read data**/
    /* Need to read the data to know the sample sizes and the number of features */
    for(k=1;k<=num_condit;k++){
   // printf("k is %d\n", k);
    sprintf(s, "%s/A%d.dat", argv[1],k);
    // argv[0] represents the program name, argv[1] means the first parameter, the data folder
    // printf("Reading %s\n", s);
    f = fopen(s, "r");
	if (f == NULL) {
		printf("Reading ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
    mm_read_mtx_array_size(f, &m0, &p);
    gsl_vector_set(M,k-1,m0);// different conditions maybe have different sample size
    }
    printf("M0 is %g.\n",gsl_vector_get(M,0));
    printf("M1 is %g.\n",gsl_vector_get(M,1));
   //// printf("M2 is %g.\n",gsl_vector_get(M,2));
    printf("*****************\n");


    /** Defination of X,Y,beta **/
    /* define X and Y once we knew sample sizes and number of features */
    m=gsl_blas_dasum(M); //m is the total number of samples under multiple conditions
    ncol_X=num_condit*p;//ncol_X is the number of columns in X
    printf("the total sample size is %d.\n",m);


    if(fused_type==1){
    m_D=p*(num_condit-1);
    }
    if(fused_type==2){
    m_D=p*(num_condit*(num_condit-1)/2);
    }

    // the number of columns of D matrix
    n_D=p*num_condit;
    gsl_matrix *D = gsl_matrix_calloc(m_D, n_D);
    gsl_matrix *Dt = gsl_matrix_calloc(n_D, m_D);
    gsl_matrix *DtD = gsl_matrix_calloc(n_D, n_D);
    double *temp_a[m_D];
    double *temp_b[n_D];
    double *temp_c[n_D];
        for(i=0;i<m_D;i++){
            temp_a[i]=(double *)malloc(n_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_b[i]=(double *)malloc(m_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_c[i]=(double *)malloc(n_D*sizeof(double));
        }

    gsl_matrix *X= gsl_matrix_calloc(m, ncol_X); // X is the microarray expression matrix
	gsl_vector *Y= gsl_vector_calloc(m);// Y is the target gene
    gsl_vector *beta = gsl_vector_calloc(ncol_X);
    gsl_vector *beta_lasso= gsl_vector_calloc(p);//beta1=beta2=beta3=beta_lasso when calculate lambda2_max given a specific lambda1
	gsl_matrix *X_lasso= gsl_matrix_calloc(m, p);
    gsl_vector *XtY    = gsl_vector_calloc(ncol_X);
    gsl_matrix *XtX    = gsl_matrix_calloc(ncol_X,ncol_X);
    gsl_vector *v    = gsl_vector_calloc(ncol_X);
    gsl_vector *X_lassotY    = gsl_vector_calloc(p);
    gsl_matrix *X_lassotX_lasso  = gsl_matrix_calloc(p,p);
    ////gsl_vector *z= gsl_vector_calloc(m_D);


     gsl_vector *Temp=gsl_vector_calloc(ncol_X);
     gsl_vector *XtX_beta= gsl_vector_calloc(ncol_X);


    int m1=gsl_vector_get(M,0);
    int m2=gsl_vector_get(M,1);
    ////int m3=gsl_vector_get(M,2);
    int m3=gsl_vector_get(M,0);
    gsl_matrix *X1= gsl_matrix_calloc(m1, p);
    gsl_matrix *X2= gsl_matrix_calloc(m2, p);
    gsl_matrix *X3= gsl_matrix_calloc(m3, p);
//    gsl_matrix *X1_new= gsl_matrix_calloc(gsl_vector_get(M,0), p);
//    gsl_matrix *X2_new= gsl_matrix_calloc(gsl_vector_get(M,1), p);
    gsl_vector *Y1= gsl_vector_calloc(m1);
    gsl_vector *Y2= gsl_vector_calloc(m2);
    gsl_vector *Y3= gsl_vector_calloc(m3);




      /** Build D matrix **/
  /************************************Build D matrix Start*****************************************************/
                    /**Build D matrix according to num_condit, lambda1 and lambda2**/

                    Build_D_matrix(D,p,num_condit,fused_type,1,1);

                    for(i=0;i<n_D;i++){
                    for(j=0;j<m_D;j++){
                    // Dt: D transpose
                    gsl_matrix_set(Dt,i,j,gsl_matrix_get(D,j,i));}
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_a,D)
                    {
                       //#pragma omp for
                    for(i=0;i<m_D;i++){
                        for(j=0;j<n_D;j++){
                            temp_a[i][j]=gsl_matrix_get(D,i,j);}
                    }
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_b,Dt)
                    {
                   // #pragma omp for
                    for(i=0;i<n_D;i++){
                        for(j=0;j<m_D;j++){
                            temp_b[i][j]=gsl_matrix_get(Dt,i,j);
                        }
                    }
                    }


                    for(i=0;i<n_D;i++){
                        for(j=0;j<n_D;j++){
                             temp_c[i][j]=0;
                             gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                            }
                     }
                  //#pragma omp parallel for schedule(static,m/N_threads) private(i,j,temp_k) shared(DtD,temp_c)
                    for(i=0;i<n_D;i++){
                        for( j=0;j<n_D;j++){
                            for(temp_k=0;temp_k<m_D;temp_k++){
                                temp_c[i][j]+=temp_b[i][temp_k]*temp_a[temp_k][j];
                            }
                            gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                        }
                    }
    /************************************Build D matrix End*****************************************************/
    /** calculate Lip_DtD **/
    Lip_DtD=PowerMethod(DtD,1000,5e-3,N_threads);
    printf("Lip_DtD is %lf\n",Lip_DtD);



    /*****define variables used in the stability selection******/
    int up_id[m1/2];//firstly only consider m1=m2=m3
    int down_id[m1/2];
    int m_up,n_up,m_down,n_down;
    m_up=m/2;
    n_up=p*num_condit;
    m_down=m_up;
    n_down=n_up;
    gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
    gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
    gsl_vector *up_Y=gsl_vector_calloc(m_up);

     gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
     gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
     gsl_vector *down_Y=gsl_vector_calloc(m_down);

    //// up_z,up_x,down_z,down_x are the variables during each round ////
    gsl_vector *up_z= gsl_vector_calloc(m_D);
    gsl_vector *up_beta= gsl_vector_calloc(n_up);
    gsl_vector *down_z= gsl_vector_calloc(m_D);
    gsl_vector *down_beta= gsl_vector_calloc(n_down);

    // store the x during Stability selection(all rounds),each row is a solution,

    /* initialize up part and down part in stability selection */
	for(i=0;i<m_up;i++)
    for(j=0;j<n_up;j++){
        gsl_matrix_set(up_X,i,j,0);
        gsl_matrix_set(down_X,i,j,0);
    }
    for(i=0;i<m_up;i++){
        gsl_vector_set(up_Y,i,0);
        gsl_vector_set(down_Y,i,0);
    }

    //// read the index of subsamples for each round /////
      gsl_matrix *SS_index=gsl_matrix_calloc(2*rounds,m1/2);
      sprintf(s, "SS_index_%d_.dat",m1);
    // SS_index the first row is the index for up part in the file
    f = fopen(s, "r");
    if (f == NULL) {
        printf("Reading ERROR: %s does not exist, exiting.\n", s);
        exit(EXIT_FAILURE);
    }
    mm_read_mtx_array_size(f, &s_m0, &s_n0);
    for (int i = 0; i < s_m0*s_n0; i++) {
        row = i % s_m0;
        col = floor(i/s_m0);
        fscanf(f, "%lf", &s_entry);
        gsl_matrix_set(SS_index, row, col, s_entry);
    }

//    printf("SS_index is:\n" );
//    for(i=0;i<s_m0;i++){
//        printf("SS_index[%d]: ",i);
//        for(j=0;j<s_n0;j++){
//            printf("%g ",gsl_matrix_get(SS_index,i,j));
//            if(j==s_n0-1) printf("\n");
//        }
//    }

	/** Read data **/
    /* read X */ //the data is stored by column
    for(k=1;k<=num_condit;k++){
        sprintf(s, "%s/A%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &m0, &p);
        for (int i = 0; i < m0*p; i++) {
            row = i % m0;
            col = floor(i/m0);
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_matrix_set(X1,row,col, entry);
                gsl_matrix_set(X,row,col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso, row, col, entry);
            }
            if(k==2){
                gsl_matrix_set(X2,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0), col, entry);
            }
//            if(k==3){
//                gsl_matrix_set(X3,row,col, entry);
//                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1),col+(k-1)*p, entry);
//                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1), col, entry);
//            }

        }
        fclose(f);

        /* Read Y */
	  sprintf(s, "%s/b%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &nrow_y,&ncol_y);
        for (int i = 0; i < nrow_y; i++) {
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_vector_set(Y1, i, entry);
                gsl_vector_set(Y, i, entry);}
            if(k==2){
                gsl_vector_set(Y2, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0), entry);}
//            if(k==3){
//                gsl_vector_set(Y3, i, entry);
//                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1), entry);}

        }
        fclose(f);

    }
            /* Precompute and cache factorizations */
     gsl_blas_dgemv(CblasTrans, 1, X, Y, 0, XtY); // XtY = X^T*Y
     gsl_vector_memcpy(Temp,XtY);
     gsl_blas_dgemv(CblasTrans, 1, X_lasso, Y, 0, X_lassotY); // XtY = X^T*Y
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X,0.0, XtX);
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X_lasso, X_lasso,0.0, X_lassotX_lasso);
//     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)/m;
     Lip_data=PowerMethod(XtX,1000,5e-3,N_threads)*2;
     Lip=Lip_data+Lip_DtD;
     printf("Lip is %lf\n",Lip);
     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)*2+1;



     /**calculate lambda1_ss and lambda2_ss**/
     /**lambda1_max=max(X^TY)**/
        for(int i=0;i<ncol_X;i++){
       gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
              gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
                  }
       lambda1_ss_max=gsl_vector_max(v)*2;
     printf(" lambda1_ss_max is: %g\n",lambda1_ss_max);
    /* generate lambda1 */
      lambda1_ss_min=lambda1_ss_max/100;//pow(0.7,num_lambda1_ss-1);
       printf(" lambda1_ss_min is: %g\n",lambda1_ss_min);
      del_log_lambda1_ss=(log10(lambda1_ss_max)-log10(lambda1_ss_min))/(num_lambda1_ss-1);
       printf(" del_log_lambda1_ss is: %g\n",del_log_lambda1_ss);
      for(i=0;i<num_lambda1_ss;i++){
        gsl_vector_set(Lambda1_ss,i,pow(10,log10(lambda1_ss_max)-del_log_lambda1_ss*i));
      }

    printf("\n" );
    printf("Lambda1_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++)
    printf("%g ",gsl_vector_get(Lambda1_ss,i));
    printf("\n");



        for(i=0;i<num_lambda1_ss;i++){
       // ProAdmmLasso(X_lasso,X_lassotX_lasso,X_lassotY,Y,beta_lasso,gsl_vector_get(Lambda1_ss,i),1,Lip_lasso,2000,N_threads);

        FISTA(X_lasso,Y, X_lassotX_lasso, X_lassotY, 2*gsl_vector_get(Lambda1_ss,i),beta_lasso,Lip_lasso);
             printf("The lasso solution is:\n");
             for(int temp_i=0;temp_i<beta_lasso->size;temp_i++){
               printf("%g ",gsl_vector_get(beta_lasso,temp_i));
             }

        for(temp_i=0;temp_i<beta_lasso->size;temp_i++){
            gsl_vector_set(beta,temp_i,gsl_vector_get(beta_lasso,temp_i));
            gsl_vector_set(beta,temp_i+p,gsl_vector_get(beta_lasso,temp_i));
            //gsl_vector_set(beta,temp_i+2*p,gsl_vector_get(beta_lasso,temp_i));
        }

        gsl_blas_dgemv(CblasNoTrans, 1, XtX, beta, 0, XtX_beta);
        gsl_vector_memcpy(Temp,XtY); /// It's very important to recopy XtY to Temp because Temp will change after calculation
        gsl_vector_sub(Temp,XtX_beta);//XtY-XtX*beta

        for(int t=0;t<(Temp->size);t++){
        temp=gsl_vector_get(Temp,t);
        gsl_vector_set(Temp,t,fabs(temp));
        //printf("Temp[%d] is %g\n",i,gsl_vector_get(Temp,t));
        }
       // printf("gsl_vector_max(Temp):%g\n",gsl_vector_max(Temp));
      lambda2_ss_max=gsl_vector_max(Temp)*2+gsl_vector_get(Lambda1_ss,i);
      printf("lambda2_ss_max is:%g\n",lambda2_ss_max);

            gsl_matrix_set(Lambda2_ss,i,0,lambda2_ss_max);
            lambda2_ss_min=lambda2_ss_max/100;//*pow(0.8,num_lambda2_ss-1);
             for(j=0;j<num_lambda2_ss;j++){
                del_log_lambda2_ss=(log10(lambda2_ss_max)-log10(lambda2_ss_min))/(num_lambda2_ss-1);
                gsl_matrix_set(Lambda2_ss,i,j,pow(10,log10(lambda2_ss_max)-del_log_lambda2_ss*j));
             }
    }


    printf("Lambda2_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++){
        printf("Lambda2[%d]: ",i);
        for(j=0;j<num_lambda2_ss;j++){
            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
            if(j==num_lambda2_ss-1) printf("\n");
        }
    }
        /// save Lambda1_SS and Lambda2_SS
         sprintf(s, "solution_SS/Lambda1_SS.dat");
         f = fopen(s, "w");
         gsl_vector_fprintf(f, Lambda1_ss, "%g");
         fclose(f);

         sprintf(s, "solution_SS/Lambda2_SS.dat");
         f = fopen(s, "w");
         gsl_matrix_fprintf(f, Lambda2_ss, "%g");
         fclose(f);















//     /**Read the lambda1_ss and lambda2_ss from the lambda1 and lambda2 calculated by lastPaper**/
//    sprintf(s, "data/Lambda1.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss; i++) {
//            fscanf(f, "%lf", &entry);
//         gsl_vector_set(Lambda1_ss,i,entry);
//        }
//    fclose(f);
//    printf("\n" );
//    printf("Lambda1_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++)
//    printf("%g ",gsl_vector_get(Lambda1_ss,i));
//    printf("\n");
//
//    sprintf(s, "data/Lambda2.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss*num_lambda2_ss; i++) {
//            row = i % num_lambda1_ss;
//            col = floor(i/num_lambda1_ss);
//            fscanf(f, "%lf", &entry);
//         gsl_matrix_set(Lambda2_ss,row,col, entry);
//        }
//    fclose(f);

//    printf("Lambda2_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++){
//        printf("Lambda2[%d]: ",i);
//        for(j=0;j<num_lambda2_ss;j++){
//            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
//            if(j==num_lambda2_ss-1) printf("\n");
//        }
//    }


    /// initialize Lip_up_temp
    for(i=0;i<rounds;i++){
        gsl_vector_set(Lip_up_temp,i,Lip);
        gsl_vector_set(Lip_down_temp,i,Lip);
    }

    /// initialize beta
    for(i=0;i<n_up;i++){
//        gsl_vector_set(up_beta,i,1.0);
 //       gsl_vector_set(down_beta,i,1.0);
//        gsl_vector_set(up_beta,i,0.0);
//        gsl_vector_set(down_beta,i,0.0);
        gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
        gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
    }

     /// initialize z
    for(int i=0;i<(up_z->size);i++){
        gsl_vector_set(up_z,i,0);
        gsl_vector_set(down_z,i,0);
       // printf("i is: %d, ThreadId=%d\n",i,omp_get_thread_num());
    }




    /// calculate Lipschitz constant
     printf("Start to calculate Lipschitz constant \n");
/******************************************************************************************/
    for (int r=0;r<rounds;r++){
         for(i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }
          for(k=1;k<=num_condit;k++){
           for(i=0;i<m1/2;i++){
                     for(j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }
                            if(k==3){

                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                    }

                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
           }

        }

                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);

                     Lip_temp=PowerMethod(up_Xtup_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_up_temp,r,Lip_temp);
                     Lip_temp=PowerMethod(down_Xtdown_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_down_temp,r,Lip_temp);


    }
     printf("End up calculating Lipschitz constant \n");
/******************************************************************************************/







/******************************************************************************************/

//  for(i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
//        lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
//    for(j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){
//        lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);


gsl_matrix *SS_beta=gsl_matrix_calloc(n_up*2,num_lambda1_ss*num_lambda2_ss*rounds);
gsl_matrix_set_zero(SS_beta);

//#pragma omp parallel for collapse(3) private(r,up_id,down_id,i,k,j,up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X,Lip,st_down,et_down,st_up,et_up,down_beta,down_z,up_beta,up_z,f,i_lambda1,j_lambda2,lambda1_ss,lambda2_ss) shared(SS_beta,n_up,num_condit,rounds,m1,SS_index,p,X1,X2,X3,Y1,Y2,Y3,Lip_up_temp,Lip_down_temp,Lip_DtD,D,Dt,DtD,Lambda1_ss,Lambda2_ss)

  for( i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
    for( j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){
         //for (r=0;r<rounds;r++){
         for (int r=0;r<1;r++){
         //
         //
         gsl_matrix *ss_beta=gsl_matrix_calloc(2,n_up);
         gsl_matrix_set_zero(ss_beta);

           // printf("The i_lambda1 is:%d, j_lambda2 is:%d, r is :%d, Thread_ID:%d\n",i_lambda1,j_lambda2,r,omp_get_thread_num());
            double lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
            double lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);
            double Lip=0.0;


            /// up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X are private variables, we need to redefine them in each thread
            gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
            gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *up_Y=gsl_vector_calloc(m_up);

            gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
            gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *down_Y=gsl_vector_calloc(m_down);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector *up_z= gsl_vector_calloc(m_D);
            gsl_vector *up_beta= gsl_vector_calloc(n_up);
            gsl_vector *down_z= gsl_vector_calloc(m_D);
            gsl_vector *down_beta= gsl_vector_calloc(n_down);

            gsl_matrix_set_zero(up_X);
            gsl_matrix_set_zero(down_X);
            gsl_matrix_set_zero(up_Xtup_X);
            gsl_matrix_set_zero(down_Xtdown_X);
            gsl_vector_set_zero(up_Y);
            gsl_vector_set_zero(down_Y);

            /// initialize beta with beta_lasso
            /*for(int i=0;i<n_up;i++){
                gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
                gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
            }*/

            /// initialize beta with zero
            gsl_vector_set_zero(up_beta);
            gsl_vector_set_zero(down_beta);
            /// initialize z
            for(int i=0;i<(up_z->size);i++){
                gsl_vector_set(up_z,i,0);
                gsl_vector_set(down_z,i,0);
            }


            printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);
            //split data into two parts: the up part and down part
           // printf("Test_id is\n");
          /// assign the data into two parts
            for(int i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }

           for(int k=1;k<=num_condit;k++){
            for(int i=0;i<m1/2;i++){
                 for(int j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }
//                            if(k==3){
//
//                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i],j));
//                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i],j));
//                        }
                    }

                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
//                  if(k==3){
//                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]));
//                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]));
//                }

           }

        }

             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);



            /********* Main AIMNet_solver loop **********/
               /**up part**/
                Lip=gsl_vector_get(Lip_up_temp,r)+Lip_DtD;
                printf("Lip_up is %lf\n",Lip);
                //st_up=omp_get_wtime();
              /********* Main ADMM solver loop **********/
              AIMNet_Solver(N_threads,up_X,up_Y,up_beta,up_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                //et_up=omp_get_wtime()-st_up;



               /**down part**/
                 Lip=gsl_vector_get(Lip_down_temp,r)+Lip_DtD;
                 printf("Lip_down is %lf\n",Lip);
                 //st_down=omp_get_wtime();
                 AIMNet_Solver(N_threads,down_X,down_Y,down_beta,down_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                 //et_down=omp_get_wtime()-st_down;

                //printf("The total threads are %d , time_down is:%15.15lf\n",N_threads, et_down);
                // printf("The total threads are %d , time_up is: %15.15lf, time_down is:%15.15lf\n",N_threads,et_up,et_down);




           for(int j=0;j<down_beta->size;j++){
               gsl_matrix_set(ss_beta,0,j,gsl_vector_get(up_beta,j));
               gsl_matrix_set(ss_beta,1,j,gsl_vector_get(down_beta,j));
               gsl_matrix_set(SS_beta,j,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(up_beta,j));
               gsl_matrix_set(SS_beta,j+down_beta->size,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(down_beta,j));
            }

          /*sprintf(s, "solution_SS/ss_beta_%d_%d_r_%d.dat", i_lambda1,j_lambda2,r);
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, ss_beta, "%g");
            fclose(f);
          */
            gsl_matrix_free(up_X);
            gsl_matrix_free(up_Xtup_X);
            gsl_vector_free(up_Y);
            gsl_matrix_free(down_X);
            gsl_matrix_free(down_Xtdown_X);
            gsl_vector_free(down_Y);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector_free(up_z);
            gsl_vector_free(up_beta);
            gsl_vector_free(down_z);
            gsl_vector_free(down_beta);

         }

    }
  }


  sprintf(s, "solution_SS/SS_beta.dat");
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, SS_beta, "%g");
            fclose(f);

 /******************************************************************************************/

 printf("the maximum number of threads is: %d\n", omp_get_max_threads());
 printf("the number of threads used in parallel is: %d\n",N_threads);



 return 0;



}






/*
Solve the differential gene regulatory networks inference problem under three conditions.
Set the number of conditions in head.h file: num_condit=3.
The program uses OpenMP for parallel implementation and the GNU Scientific Library(GSL) for math.

Run the program in parallel:
    Use "export OMP_NUM_THREADS=num_threads" to tell the program the number of threads.
    For example:
        In the command line:
            export OMP_NUM_THREADS=16
        In the code:
            N_threads=omp_get_max_threads()-1;// which means the number of threads used in parallel is 15 (the total threads number-1).

Compile the code in local:
    gcc -Wall -std=c99 -I/GSL_path/include -c mmio.c -o mmio.o
    gcc -fopenmp -Wall -std=c99 -g -I/GSL_path/include -c simu_3condits.c -o simu_3condits.o
    gcc -fopenmp -Wall -std=c99 -g -L/GSL_path/lib  simu_3condits.o mmio.o -o simu_3condits -lgsl -lgslcblas -lm

    For example:
   	gcc -Wall -std=c99 -I/usr/local/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/usr/local/include -c simu_3condits.c -o simu_3condits.o
	gcc -fopenmp -Wall -std=c99 -g -L/usr/local/lib  simu_3condits.o mmio.o -o simu_3condits -lgsl -lgslcblas -lm

Run program:
    // export the threads used in parallel
    // demo_data: the example data, 3 is the number of tissues
    export OMP_NUM_THREADS=5
    ./simu_3condits demo_data 3

*/

int simu_3condits(char *argv[])
{

    int num_condit;
    long conv=strtol(argv[2],NULL,10); // the second parameter is the number of conditions
    num_condit=conv;

    FILE *f;
    char s[40];
    double entry,s_entry,Lip_temp;
    //double st_up,et_up,st_down,et_down;
    double Lip_data,Lip_DtD;
    double Lip,Lip_lasso,temp_i,temp;

    int i,j,k,m,m0,p,ncol_X,m_D,n_D;
    gsl_vector  *Lip_up_temp=gsl_vector_calloc(rounds);
    gsl_vector  *Lip_down_temp=gsl_vector_calloc(rounds);
    int temp_k=0;
    int ncol_y,nrow_y;
    int row,col;
    int s_m0,s_n0;
    int i_lambda1,j_lambda2;
    //int N_threads=1; // the number of threads
    int N_threads=omp_get_max_threads()-1;

   // double lambda1_ss,lambda2_ss;
    double lambda1_ss_max,lambda1_ss_min,lambda2_ss_max,lambda2_ss_min;
    double del_log_lambda1_ss,del_log_lambda2_ss;
    gsl_vector *Lambda1_ss=gsl_vector_calloc(num_lambda1_ss);
    gsl_matrix *Lambda2_ss=gsl_matrix_calloc(num_lambda1_ss,num_lambda2_ss);
    gsl_vector *M= gsl_vector_calloc(num_condit);//M stores the number of samples under each condition
    // M[0] is the first condition
    //gsl_vector *T=gsl_vector_calloc(1);
    mkdir("solution_SS",0700);

    /**Read data**/
    /* Need to read the data to know the sample sizes and the number of features */
    for(k=1;k<=num_condit;k++){
   // printf("k is %d\n", k);
    sprintf(s, "%s/A%d.dat", argv[1],k);
    // argv[0] represents the program name, argv[1] means the first parameter
    // printf("Reading %s\n", s);
    f = fopen(s, "r");
	if (f == NULL) {
		printf("Reading ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
    mm_read_mtx_array_size(f, &m0, &p);
    gsl_vector_set(M,k-1,m0);// different conditions maybe have different sample size
    }
    printf("M0 is %g.\n",gsl_vector_get(M,0));
    printf("M1 is %g.\n",gsl_vector_get(M,1));
    //printf("M2 is %g.\n",gsl_vector_get(M,2));
    for(k=1;k<=num_condit;k++){
       printf("The sample size in condition[%d] is %g.\n",k,gsl_vector_get(M,k-1));
    }
    printf("*****************\n");


    /** Defination of X,Y,beta **/
    /* define X and Y once we knew sample sizes and number of features */
    m=gsl_blas_dasum(M); //m is the total number of samples under multiple conditions
    ncol_X=num_condit*p;//ncol_X is the number of columns in X
    printf("the total sample size is %d.\n",m);


    if(fused_type==1){
    m_D=p*(num_condit-1);
    }
    if(fused_type==2){
    m_D=p*(num_condit*(num_condit-1)/2);
    }

    // the number of columns of D matrix
    n_D=p*num_condit;
    gsl_matrix *D = gsl_matrix_calloc(m_D, n_D);
    gsl_matrix *Dt = gsl_matrix_calloc(n_D, m_D);
    gsl_matrix *DtD = gsl_matrix_calloc(n_D, n_D);
    double *temp_a[m_D];
    double *temp_b[n_D];
    double *temp_c[n_D];
        for(i=0;i<m_D;i++){
            temp_a[i]=(double *)malloc(n_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_b[i]=(double *)malloc(m_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_c[i]=(double *)malloc(n_D*sizeof(double));
        }


    gsl_matrix *X= gsl_matrix_calloc(m, ncol_X); // X is the microarray expression matrix
	gsl_vector *Y= gsl_vector_calloc(m);// Y is the target gene
    gsl_vector *beta = gsl_vector_calloc(ncol_X);
    gsl_vector *beta_lasso= gsl_vector_calloc(p);//beta1=beta2=beta3=beta_lasso when calculate lambda2_max given a specific lambda1
	gsl_matrix *X_lasso= gsl_matrix_calloc(m, p);
    gsl_vector *XtY    = gsl_vector_calloc(ncol_X);
    gsl_matrix *XtX    = gsl_matrix_calloc(ncol_X,ncol_X);
    gsl_vector *v    = gsl_vector_calloc(ncol_X);
    gsl_vector *X_lassotY    = gsl_vector_calloc(p);
    gsl_matrix *X_lassotX_lasso  = gsl_matrix_calloc(p,p);
    ////gsl_vector *z= gsl_vector_calloc(m_D);


     gsl_vector *Temp=gsl_vector_calloc(ncol_X);
     gsl_vector *XtX_beta= gsl_vector_calloc(ncol_X);


//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
    int m1=gsl_vector_get(M,0);
    int m2=gsl_vector_get(M,1);
    int m3=gsl_vector_get(M,2);
    ///int m3=gsl_vector_get(M,0);
    gsl_matrix *X1= gsl_matrix_calloc(m1, p);
    gsl_matrix *X2= gsl_matrix_calloc(m2, p);
    gsl_matrix *X3= gsl_matrix_calloc(m3, p);

    gsl_vector *Y1= gsl_vector_calloc(m1);
    gsl_vector *Y2= gsl_vector_calloc(m2);
    gsl_vector *Y3= gsl_vector_calloc(m3);
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//




      /** Build D matrix **/
  /************************************Build D matrix Start*****************************************************/
                    /**Build D matrix according to num_condit, lambda1 and lambda2**/

                    Build_D_matrix(D,p,num_condit,fused_type,1,1);

                    for(i=0;i<n_D;i++){
                    for(j=0;j<m_D;j++){
                    // Dt: D transpose
                    gsl_matrix_set(Dt,i,j,gsl_matrix_get(D,j,i));}
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_a,D)
                    {
                       //#pragma omp for
                    for(i=0;i<m_D;i++){
                        for(j=0;j<n_D;j++){
                            temp_a[i][j]=gsl_matrix_get(D,i,j);}
                    }
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_b,Dt)
                    {
                   // #pragma omp for
                    for(i=0;i<n_D;i++){
                        for(j=0;j<m_D;j++){
                            temp_b[i][j]=gsl_matrix_get(Dt,i,j);
                        }
                    }
                    }


                    for(i=0;i<n_D;i++){
                        for(j=0;j<n_D;j++){
                             temp_c[i][j]=0;
                             gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                            }
                     }
                  //#pragma omp parallel for schedule(static,m/N_threads) private(i,j,temp_k) shared(DtD,temp_c)
                    for(i=0;i<n_D;i++){
                        for( j=0;j<n_D;j++){
                            for(temp_k=0;temp_k<m_D;temp_k++){
                                temp_c[i][j]+=temp_b[i][temp_k]*temp_a[temp_k][j];
                            }
                            gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                        }
                    }
    /************************************Build D matrix End*****************************************************/
    /** calculate Lip_DtD **/
    Lip_DtD=PowerMethod(DtD,1000,5e-3,N_threads);
    printf("Lip_DtD is %lf\n",Lip_DtD);



    /*****define variables used in the stability selection******/
    int up_id[m1/2];//firstly only consider m1=m2=m3
    int down_id[m1/2];
    int m_up,n_up,m_down,n_down;
    m_up=m/2;
    n_up=p*num_condit;
    m_down=m_up;
    n_down=n_up;
    gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
    gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
    gsl_vector *up_Y=gsl_vector_calloc(m_up);

     gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
     gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
     gsl_vector *down_Y=gsl_vector_calloc(m_down);

    //// up_z,up_x,down_z,down_x are the variables during each round ////
    gsl_vector *up_z= gsl_vector_calloc(m_D);
    gsl_vector *up_beta= gsl_vector_calloc(n_up);
    gsl_vector *down_z= gsl_vector_calloc(m_D);
    gsl_vector *down_beta= gsl_vector_calloc(n_down);


    /* initialize up part and down part in stability selection */
	for(i=0;i<m_up;i++)
    for(j=0;j<n_up;j++){
        gsl_matrix_set(up_X,i,j,0);
        gsl_matrix_set(down_X,i,j,0);
    }
    for(i=0;i<m_up;i++){
        gsl_vector_set(up_Y,i,0);
        gsl_vector_set(down_Y,i,0);
    }

    //// read the index of subsamples for each round /////
      gsl_matrix *SS_index=gsl_matrix_calloc(2*rounds,m1/2);
      sprintf(s, "SS_index_%d_.dat",m1);
    // SS_index the first row is the index for up part in the file
    f = fopen(s, "r");
    if (f == NULL) {
        printf("Reading ERROR: %s does not exist, exiting.\n", s);
        exit(EXIT_FAILURE);
    }
    mm_read_mtx_array_size(f, &s_m0, &s_n0);
    for (int i = 0; i < s_m0*s_n0; i++) {
        row = i % s_m0;
        col = floor(i/s_m0);
        fscanf(f, "%lf", &s_entry);
        gsl_matrix_set(SS_index, row, col, s_entry);
    }

//    printf("SS_index is:\n" );
//    for(i=0;i<s_m0;i++){
//        printf("SS_index[%d]: ",i);
//        for(j=0;j<s_n0;j++){
//            printf("%g ",gsl_matrix_get(SS_index,i,j));
//            if(j==s_n0-1) printf("\n");
//        }
//    }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
	/** Read data **/
    /* read X */ //the data is stored by column
    for(k=1;k<=num_condit;k++){
        sprintf(s, "%s/A%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &m0, &p);
        for (int i = 0; i < m0*p; i++) {
            row = i % m0;
            col = floor(i/m0);
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_matrix_set(X1,row,col, entry);
                gsl_matrix_set(X,row,col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso, row, col, entry);
            }
            if(k==2){
                gsl_matrix_set(X2,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0), col, entry);
            }
            if(k==3){
                gsl_matrix_set(X3,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1), col, entry);
            }

        }
        fclose(f);

        /* Read Y */
	  sprintf(s, "%s/b%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &nrow_y,&ncol_y);
        for (int i = 0; i < nrow_y; i++) {
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_vector_set(Y1, i, entry);
                gsl_vector_set(Y, i, entry);}
            if(k==2){
                gsl_vector_set(Y2, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0), entry);}
            if(k==3){
                gsl_vector_set(Y3, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1), entry);}
        }
        fclose(f);

    }
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//



            /* Precompute and cache factorizations */
     gsl_blas_dgemv(CblasTrans, 1, X, Y, 0, XtY); // XtY = X^T*Y
     gsl_vector_memcpy(Temp,XtY);
     gsl_blas_dgemv(CblasTrans, 1, X_lasso, Y, 0, X_lassotY); // XtY = X^T*Y
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X,0.0, XtX);
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X_lasso, X_lasso,0.0, X_lassotX_lasso);
//     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)/m;
     Lip_data=PowerMethod(XtX,1000,5e-3,N_threads)*2;
     Lip=Lip_data+Lip_DtD;
     printf("Lip is %lf\n",Lip);
     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)*2+1;



     /**calculate lambda1_ss and lambda2_ss**/
     /**lambda1_max=max(X^TY)**/
        for(int i=0;i<ncol_X;i++){
       gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
              gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
                  }
       lambda1_ss_max=gsl_vector_max(v)*2;
     printf(" lambda1_ss_max is: %g\n",lambda1_ss_max);
    /* generate lambda1 */
      lambda1_ss_min=lambda1_ss_max/100;//pow(0.7,num_lambda1_ss-1);
       printf(" lambda1_ss_min is: %g\n",lambda1_ss_min);
      del_log_lambda1_ss=(log10(lambda1_ss_max)-log10(lambda1_ss_min))/(num_lambda1_ss-1);
       printf(" del_log_lambda1_ss is: %g\n",del_log_lambda1_ss);
      for(i=0;i<num_lambda1_ss;i++){
        gsl_vector_set(Lambda1_ss,i,pow(10,log10(lambda1_ss_max)-del_log_lambda1_ss*i));
      }

    printf("\n" );
    printf("Lambda1_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++)
    printf("%g ",gsl_vector_get(Lambda1_ss,i));
    printf("\n");



        for(i=0;i<num_lambda1_ss;i++){
       // ProAdmmLasso(X_lasso,X_lassotX_lasso,X_lassotY,Y,beta_lasso,gsl_vector_get(Lambda1_ss,i),1,Lip_lasso,2000,N_threads);

        FISTA(X_lasso,Y, X_lassotX_lasso, X_lassotY, 2*gsl_vector_get(Lambda1_ss,i),beta_lasso,Lip_lasso);
             printf("The lasso solution is:\n");
             for(int temp_i=0;temp_i<beta_lasso->size;temp_i++){
               printf("%g ",gsl_vector_get(beta_lasso,temp_i));
             }

        for(temp_i=0;temp_i<beta_lasso->size;temp_i++){
            gsl_vector_set(beta,temp_i,gsl_vector_get(beta_lasso,temp_i));
            gsl_vector_set(beta,temp_i+p,gsl_vector_get(beta_lasso,temp_i));
            //gsl_vector_set(beta,temp_i+2*p,gsl_vector_get(beta_lasso,temp_i));
        }

        gsl_blas_dgemv(CblasNoTrans, 1, XtX, beta, 0, XtX_beta);
        gsl_vector_memcpy(Temp,XtY); /// It's very important to recopy XtY to Temp because Temp will change after calculation
        gsl_vector_sub(Temp,XtX_beta);//XtY-XtX*beta

        for(int t=0;t<(Temp->size);t++){
        temp=gsl_vector_get(Temp,t);
        gsl_vector_set(Temp,t,fabs(temp));
        //printf("Temp[%d] is %g\n",i,gsl_vector_get(Temp,t));
        }
       // printf("gsl_vector_max(Temp):%g\n",gsl_vector_max(Temp));
      lambda2_ss_max=gsl_vector_max(Temp)*2+gsl_vector_get(Lambda1_ss,i);
      printf("lambda2_ss_max is:%g\n",lambda2_ss_max);

            gsl_matrix_set(Lambda2_ss,i,0,lambda2_ss_max);
            lambda2_ss_min=lambda2_ss_max/100;//*pow(0.8,num_lambda2_ss-1);
             for(j=0;j<num_lambda2_ss;j++){
                del_log_lambda2_ss=(log10(lambda2_ss_max)-log10(lambda2_ss_min))/(num_lambda2_ss-1);
                gsl_matrix_set(Lambda2_ss,i,j,pow(10,log10(lambda2_ss_max)-del_log_lambda2_ss*j));
             }
    }


    printf("Lambda2_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++){
        printf("Lambda2[%d]: ",i);
        for(j=0;j<num_lambda2_ss;j++){
            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
            if(j==num_lambda2_ss-1) printf("\n");
        }
    }
        /// save Lambda1_SS and Lambda2_SS
         sprintf(s, "solution_SS/Lambda1_SS.dat");
         f = fopen(s, "w");
         gsl_vector_fprintf(f, Lambda1_ss, "%g");
         fclose(f);

         sprintf(s, "solution_SS/Lambda2_SS.dat");
         f = fopen(s, "w");
         gsl_matrix_fprintf(f, Lambda2_ss, "%g");
         fclose(f);



//     /**Read the lambda1_ss and lambda2_ss from the lambda1 and lambda2 calculated by lastPaper**/
//    sprintf(s, "data/Lambda1.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss; i++) {
//            fscanf(f, "%lf", &entry);
//         gsl_vector_set(Lambda1_ss,i,entry);
//        }
//    fclose(f);
//    printf("\n" );
//    printf("Lambda1_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++)
//    printf("%g ",gsl_vector_get(Lambda1_ss,i));
//    printf("\n");
//
//    sprintf(s, "data/Lambda2.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss*num_lambda2_ss; i++) {
//            row = i % num_lambda1_ss;
//            col = floor(i/num_lambda1_ss);
//            fscanf(f, "%lf", &entry);
//         gsl_matrix_set(Lambda2_ss,row,col, entry);
//        }
//    fclose(f);

//    printf("Lambda2_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++){
//        printf("Lambda2[%d]: ",i);
//        for(j=0;j<num_lambda2_ss;j++){
//            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
//            if(j==num_lambda2_ss-1) printf("\n");
//        }
//    }


    /// initialize Lip_up_temp
    for(i=0;i<rounds;i++){
        gsl_vector_set(Lip_up_temp,i,Lip);
        gsl_vector_set(Lip_down_temp,i,Lip);
    }

    /// initialize beta
    for(i=0;i<n_up;i++){
//        gsl_vector_set(up_beta,i,1.0);
 //       gsl_vector_set(down_beta,i,1.0);
//        gsl_vector_set(up_beta,i,0.0);
//        gsl_vector_set(down_beta,i,0.0);
        gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
        gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
    }

     /// initialize z
    for(int i=0;i<(up_z->size);i++){
        gsl_vector_set(up_z,i,0);
        gsl_vector_set(down_z,i,0);
       // printf("i is: %d, ThreadId=%d\n",i,omp_get_thread_num());
    }




    /// calculate Lipschitz constant
     printf("Start to calculate Lipschitz constant \n");
/******************************************************************************************/
    for (int r=0;r<rounds;r++){
         for(i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }

          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
          for(k=1;k<=num_condit;k++){
           for(i=0;i<m1/2;i++){
                     for(j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }
                            if(k==3){

                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                    }

                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
           }

        }
          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);

                     Lip_temp=PowerMethod(up_Xtup_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_up_temp,r,Lip_temp);
                     Lip_temp=PowerMethod(down_Xtdown_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_down_temp,r,Lip_temp);


    }
     printf("End up calculating Lipschitz constant \n");
/******************************************************************************************/
/******************************************************************************************/

//  for(i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
//        lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
//    for(j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){
//        lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);

gsl_matrix *SS_beta=gsl_matrix_calloc(n_up*2,num_lambda1_ss*num_lambda2_ss*rounds);
gsl_matrix_set_zero(SS_beta);

//#pragma omp parallel for collapse(3) private(r,up_id,down_id,i,k,j,up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X,Lip,st_down,et_down,st_up,et_up,down_beta,down_z,up_beta,up_z,f,i_lambda1,j_lambda2,lambda1_ss,lambda2_ss) shared(SS_beta,n_up,num_condit,rounds,m1,SS_index,p,X1,X2,X3,Y1,Y2,Y3,Lip_up_temp,Lip_down_temp,Lip_DtD,D,Dt,DtD,Lambda1_ss,Lambda2_ss)

  for( i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
    for( j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){

//    for( i_lambda1=0;i_lambda1<1;i_lambda1++){
//    for( j_lambda2=0;j_lambda2<1;j_lambda2++){
         //for (r=0;r<rounds;r++){
         for (int r=0;r<1;r++){
         //
         //
         gsl_matrix *ss_beta=gsl_matrix_calloc(2,n_up);
         gsl_matrix_set_zero(ss_beta);

           // printf("The i_lambda1 is:%d, j_lambda2 is:%d, r is :%d, Thread_ID:%d\n",i_lambda1,j_lambda2,r,omp_get_thread_num());
            double lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
            double lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);
            double Lip=0.0;


            /// up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X are private variables, we need to redefine them in each thread
            gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
            gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *up_Y=gsl_vector_calloc(m_up);

            gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
            gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *down_Y=gsl_vector_calloc(m_down);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector *up_z= gsl_vector_calloc(m_D);
            gsl_vector *up_beta= gsl_vector_calloc(n_up);
            gsl_vector *down_z= gsl_vector_calloc(m_D);
            gsl_vector *down_beta= gsl_vector_calloc(n_down);

            gsl_matrix_set_zero(up_X);
            gsl_matrix_set_zero(down_X);
            gsl_matrix_set_zero(up_Xtup_X);
            gsl_matrix_set_zero(down_Xtdown_X);
            gsl_vector_set_zero(up_Y);
            gsl_vector_set_zero(down_Y);

            /// initialize beta with beta_lasso
            /*for(int i=0;i<n_up;i++){
                gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
                gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
            }*/

            /// initialize beta with zero
            gsl_vector_set_zero(up_beta);
            gsl_vector_set_zero(down_beta);
            /// initialize z
            for(int i=0;i<(up_z->size);i++){
                gsl_vector_set(up_z,i,0);
                gsl_vector_set(down_z,i,0);
            }


            printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);
            //split data into two parts: the up part and down part
           // printf("Test_id is\n");
          /// assign the data into two parts
            for(int i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }
printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
           for(int k=1;k<=num_condit;k++){
            for(int i=0;i<m1/2;i++){
                 for(int j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }

                            if(k==3){
                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                    }
    printf("i[%d],j[%d]\n",i,j);
                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }

           }

        }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);



            /********* Main AIMNet_solver loop **********/
               /**up part**/
                Lip=gsl_vector_get(Lip_up_temp,r)+Lip_DtD;
                printf("Lip_up is %lf\n",Lip);
                //st_up=omp_get_wtime();
              /********* Main ADMM solver loop **********/
              AIMNet_Solver(N_threads,up_X,up_Y,up_beta,up_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                   // et_up=omp_get_wtime()-st_up;



               /**down part**/
                 Lip=gsl_vector_get(Lip_down_temp,r)+Lip_DtD;
                 printf("Lip_down is %lf\n",Lip);
                // st_down=omp_get_wtime();
                 AIMNet_Solver(N_threads,down_X,down_Y,down_beta,down_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                // et_down=omp_get_wtime()-st_down;

                //printf("The total threads are %d , time_down is:%15.15lf\n",N_threads, et_down);
                //printf("The total threads are %d , time_up is: %15.15lf, time_down is:%15.15lf\n",N_threads,et_up,et_down);




           for(int j=0;j<down_beta->size;j++){
               gsl_matrix_set(ss_beta,0,j,gsl_vector_get(up_beta,j));
               gsl_matrix_set(ss_beta,1,j,gsl_vector_get(down_beta,j));
               gsl_matrix_set(SS_beta,j,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(up_beta,j));
               gsl_matrix_set(SS_beta,j+down_beta->size,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(down_beta,j));
            }

          /*sprintf(s, "solution_SS/ss_beta_%d_%d_r_%d.dat", i_lambda1,j_lambda2,r);
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, ss_beta, "%g");
            fclose(f);
          */
            gsl_matrix_free(up_X);
            gsl_matrix_free(up_Xtup_X);
            gsl_vector_free(up_Y);
            gsl_matrix_free(down_X);
            gsl_matrix_free(down_Xtdown_X);
            gsl_vector_free(down_Y);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector_free(up_z);
            gsl_vector_free(up_beta);
            gsl_vector_free(down_z);
            gsl_vector_free(down_beta);

         }

    }
  }


  sprintf(s, "solution_SS/SS_beta.dat");
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, SS_beta, "%g");
            fclose(f);

 /******************************************************************************************/

 printf("the maximum number of threads is: %d\n", omp_get_max_threads());
 printf("the number of threads used in parallel is: %d\n",N_threads);



 return 0;



}








/*
Solve the differential gene regulatory networks inference problem under three conditions.
Set the number of conditions in head.h file: num_condit=4.
The program uses OpenMP for parallel implementation and the GNU Scientific Library(GSL) for math.

Run the program in parallel:
    Use "export OMP_NUM_THREADS=num_threads" to tell the program the number of threads.
    For example:
        In the command line:
            export OMP_NUM_THREADS=16
        In the code:
            N_threads=omp_get_max_threads()-1;// which means the number of threads used in parallel is 15 (the total threads number-1).

Compile the code in local:
    gcc -Wall -std=c99 -I/GSL_path/include -c mmio.c -o mmio.o
    gcc -fopenmp -Wall -std=c99 -g -I/GSL_path/include -c simu_4condits.c -o simu_4condits.o
    gcc -fopenmp -Wall -std=c99 -g -L/GSL_path/lib  simu_4condits.o mmio.o -o simu_4condits -lgsl -lgslcblas -lm

    For example:
   	gcc -Wall -std=c99 -I/usr/local/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/usr/local/include -c simu_4condits.c -o simu_4condits.o
	gcc -fopenmp -Wall -std=c99 -g -L/usr/local/lib  simu_4condits.o mmio.o -o simu_4condits -lgsl -lgslcblas -lm

Run program:
    // export the threads used in parallel
    // demo_data: the example data, 4 is the number of tissues
    export OMP_NUM_THREADS=5
    ./simu_4condits demo_data 4

*/

int simu_4condits(char *argv[])
{
    int num_condit;
    long conv=strtol(argv[2],NULL,10); // the second parameter is the number of conditions
    num_condit=conv;

    FILE *f;
    char s[40];
    double entry,s_entry,Lip_temp;
    //double st_up,et_up,st_down,et_down;
    double Lip_data,Lip_DtD;
    double Lip,Lip_lasso,temp_i,temp;

    int i,j,k,m,m0,p,ncol_X,m_D,n_D;
    gsl_vector  *Lip_up_temp=gsl_vector_calloc(rounds);
    gsl_vector  *Lip_down_temp=gsl_vector_calloc(rounds);
    int temp_k=0;
    int ncol_y,nrow_y;
    int row,col;
    int s_m0,s_n0;
    int i_lambda1,j_lambda2;
    //int N_threads=1;
    int N_threads=omp_get_max_threads()-1;

    //double lambda1_ss,lambda2_ss;
    double lambda1_ss_max,lambda1_ss_min,lambda2_ss_max,lambda2_ss_min;
    double del_log_lambda1_ss,del_log_lambda2_ss;
    gsl_vector *Lambda1_ss=gsl_vector_calloc(num_lambda1_ss);
    gsl_matrix *Lambda2_ss=gsl_matrix_calloc(num_lambda1_ss,num_lambda2_ss);
    gsl_vector *M= gsl_vector_calloc(num_condit);//M stores the number of samples under each condition
    // M[0] is the first condition
    //gsl_vector *T=gsl_vector_calloc(1);
    mkdir("solution_SS",0700);

    /**Read data**/
    /* Need to read the data to know the sample sizes and the number of features */
    for(k=1;k<=num_condit;k++){
   // printf("k is %d\n", k);
    sprintf(s, "%s/A%d.dat", argv[1],k);
    // argv[0] represents the program name, argv[1] means the first parameter
    // printf("Reading %s\n", s);
    f = fopen(s, "r");
	if (f == NULL) {
		printf("Reading ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
    mm_read_mtx_array_size(f, &m0, &p);
    gsl_vector_set(M,k-1,m0);// different conditions maybe have different sample size
    }
    printf("M0 is %g.\n",gsl_vector_get(M,0));
    printf("M1 is %g.\n",gsl_vector_get(M,1));
    printf("M3 is %g.\n",gsl_vector_get(M,3));
    //printf("M2 is %g.\n",gsl_vector_get(M,2));
    for(k=1;k<=num_condit;k++){
       printf("The sample size in condition[%d] is %g.\n",k,gsl_vector_get(M,k-1));
    }
    printf("*****************\n");


    /** Defination of X,Y,beta **/
    /* define X and Y once we knew sample sizes and number of features */
    m=gsl_blas_dasum(M); //m is the total number of samples under multiple conditions
    ncol_X=num_condit*p;//ncol_X is the number of columns in X
    printf("the total sample size is %d.\n",m);


    if(fused_type==1){
    m_D=p*(num_condit-1);
    }
    if(fused_type==2){
    m_D=p*(num_condit*(num_condit-1)/2);
    }

    // the number of columns of D matrix
    n_D=p*num_condit;
    gsl_matrix *D = gsl_matrix_calloc(m_D, n_D);
    gsl_matrix *Dt = gsl_matrix_calloc(n_D, m_D);
    gsl_matrix *DtD = gsl_matrix_calloc(n_D, n_D);
    double *temp_a[m_D];
    double *temp_b[n_D];
    double *temp_c[n_D];
        for(i=0;i<m_D;i++){
            temp_a[i]=(double *)malloc(n_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_b[i]=(double *)malloc(m_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_c[i]=(double *)malloc(n_D*sizeof(double));
        }


    gsl_matrix *X= gsl_matrix_calloc(m, ncol_X); // X is the microarray expression matrix
	gsl_vector *Y= gsl_vector_calloc(m);// Y is the target gene
    gsl_vector *beta = gsl_vector_calloc(ncol_X);
    gsl_vector *beta_lasso= gsl_vector_calloc(p);//beta1=beta2=beta3=beta_lasso when calculate lambda2_max given a specific lambda1
	gsl_matrix *X_lasso= gsl_matrix_calloc(m, p);
    gsl_vector *XtY    = gsl_vector_calloc(ncol_X);
    gsl_matrix *XtX    = gsl_matrix_calloc(ncol_X,ncol_X);
    gsl_vector *v    = gsl_vector_calloc(ncol_X);
    gsl_vector *X_lassotY    = gsl_vector_calloc(p);
    gsl_matrix *X_lassotX_lasso  = gsl_matrix_calloc(p,p);
    ////gsl_vector *z= gsl_vector_calloc(m_D);


     gsl_vector *Temp=gsl_vector_calloc(ncol_X);
     gsl_vector *XtX_beta= gsl_vector_calloc(ncol_X);


//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
    int m1=gsl_vector_get(M,0);
    int m2=gsl_vector_get(M,1);
    int m3=gsl_vector_get(M,2);
    int m4=gsl_vector_get(M,3);
    gsl_matrix *X1= gsl_matrix_calloc(m1, p);
    gsl_matrix *X2= gsl_matrix_calloc(m2, p);
    gsl_matrix *X3= gsl_matrix_calloc(m3, p);
    gsl_matrix *X4= gsl_matrix_calloc(m4, p);

    gsl_vector *Y1= gsl_vector_calloc(m1);
    gsl_vector *Y2= gsl_vector_calloc(m2);
    gsl_vector *Y3= gsl_vector_calloc(m3);
    gsl_vector *Y4= gsl_vector_calloc(m4);
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//




      /** Build D matrix **/
  /************************************Build D matrix Start*****************************************************/
                    /**Build D matrix according to num_condit, lambda1 and lambda2**/

                    Build_D_matrix(D,p,num_condit,fused_type,1,1);

                    for(i=0;i<n_D;i++){
                    for(j=0;j<m_D;j++){
                    // Dt: D transpose
                    gsl_matrix_set(Dt,i,j,gsl_matrix_get(D,j,i));}
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_a,D)
                    {
                       //#pragma omp for
                    for(i=0;i<m_D;i++){
                        for(j=0;j<n_D;j++){
                            temp_a[i][j]=gsl_matrix_get(D,i,j);}
                    }
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_b,Dt)
                    {
                   // #pragma omp for
                    for(i=0;i<n_D;i++){
                        for(j=0;j<m_D;j++){
                            temp_b[i][j]=gsl_matrix_get(Dt,i,j);
                        }
                    }
                    }


                    for(i=0;i<n_D;i++){
                        for(j=0;j<n_D;j++){
                             temp_c[i][j]=0;
                             gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                            }
                     }
                  //#pragma omp parallel for schedule(static,m/N_threads) private(i,j,temp_k) shared(DtD,temp_c)
                    for(i=0;i<n_D;i++){
                        for( j=0;j<n_D;j++){
                            for(temp_k=0;temp_k<m_D;temp_k++){
                                temp_c[i][j]+=temp_b[i][temp_k]*temp_a[temp_k][j];
                            }
                            gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                        }
                    }
    /************************************Build D matrix End*****************************************************/
    /** calculate Lip_DtD **/
    Lip_DtD=PowerMethod(DtD,1000,5e-3,N_threads);
    printf("Lip_DtD is %lf\n",Lip_DtD);



    /*****define variables used in the stability selection******/
    int up_id[m1/2];//firstly only consider m1=m2=m3
    int down_id[m1/2];
    int m_up,n_up,m_down,n_down;
    m_up=m/2;
    n_up=p*num_condit;
    m_down=m_up;
    n_down=n_up;
    gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
    gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
    gsl_vector *up_Y=gsl_vector_calloc(m_up);

     gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
     gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
     gsl_vector *down_Y=gsl_vector_calloc(m_down);

    //// up_z,up_x,down_z,down_x are the variables during each round ////
    gsl_vector *up_z= gsl_vector_calloc(m_D);
    gsl_vector *up_beta= gsl_vector_calloc(n_up);
    gsl_vector *down_z= gsl_vector_calloc(m_D);
    gsl_vector *down_beta= gsl_vector_calloc(n_down);


    /* initialize up part and down part in stability selection */
	for(i=0;i<m_up;i++)
    for(j=0;j<n_up;j++){
        gsl_matrix_set(up_X,i,j,0);
        gsl_matrix_set(down_X,i,j,0);
    }
    for(i=0;i<m_up;i++){
        gsl_vector_set(up_Y,i,0);
        gsl_vector_set(down_Y,i,0);
    }

    //// read the index of subsamples for each round /////
      gsl_matrix *SS_index=gsl_matrix_calloc(2*rounds,m1/2);
      sprintf(s, "SS_index_%d_.dat",m1);
    // SS_index the first row is the index for up part in the file
    f = fopen(s, "r");
    if (f == NULL) {
        printf("Reading ERROR: %s does not exist, exiting.\n", s);
        exit(EXIT_FAILURE);
    }
    mm_read_mtx_array_size(f, &s_m0, &s_n0);
    for (int i = 0; i < s_m0*s_n0; i++) {
        row = i % s_m0;
        col = floor(i/s_m0);
        fscanf(f, "%lf", &s_entry);
        gsl_matrix_set(SS_index, row, col, s_entry);
    }

//    printf("SS_index is:\n" );
//    for(i=0;i<s_m0;i++){
//        printf("SS_index[%d]: ",i);
//        for(j=0;j<s_n0;j++){
//            printf("%g ",gsl_matrix_get(SS_index,i,j));
//            if(j==s_n0-1) printf("\n");
//        }
//    }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
	/** Read data **/
    /* read X */ //the data is stored by column
    for(k=1;k<=num_condit;k++){
        sprintf(s, "%s/A%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &m0, &p);
        for (int i = 0; i < m0*p; i++) {
            row = i % m0;
            col = floor(i/m0);
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_matrix_set(X1,row,col, entry);
                gsl_matrix_set(X,row,col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso, row, col, entry);
            }
            if(k==2){
                gsl_matrix_set(X2,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0), col, entry);
            }
            if(k==3){
                gsl_matrix_set(X3,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1), col, entry);
            }
              if(k==4){
                gsl_matrix_set(X4,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2), col, entry);
            }

        }
        fclose(f);

        /* Read Y */
	  sprintf(s, "%s/b%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &nrow_y,&ncol_y);
        for (int i = 0; i < nrow_y; i++) {
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_vector_set(Y1, i, entry);
                gsl_vector_set(Y, i, entry);}
            if(k==2){
                gsl_vector_set(Y2, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0), entry);}
            if(k==3){
                gsl_vector_set(Y3, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1), entry);}
            if(k==4){
                gsl_vector_set(Y4, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2), entry);}
        }
        fclose(f);

    }
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//



            /* Precompute and cache factorizations */
     gsl_blas_dgemv(CblasTrans, 1, X, Y, 0, XtY); // XtY = X^T*Y
     gsl_vector_memcpy(Temp,XtY);
     gsl_blas_dgemv(CblasTrans, 1, X_lasso, Y, 0, X_lassotY); // XtY = X^T*Y
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X,0.0, XtX);
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X_lasso, X_lasso,0.0, X_lassotX_lasso);
//     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)/m;
     Lip_data=PowerMethod(XtX,1000,5e-3,N_threads)*2;
     Lip=Lip_data+Lip_DtD;
     printf("Lip is %lf\n",Lip);
     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)*2+1;



     /**calculate lambda1_ss and lambda2_ss**/
     /**lambda1_max=max(X^TY)**/
        for(int i=0;i<ncol_X;i++){
       gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
              gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
                  }
       lambda1_ss_max=gsl_vector_max(v)*2;
     printf(" lambda1_ss_max is: %g\n",lambda1_ss_max);
    /* generate lambda1 */
      lambda1_ss_min=lambda1_ss_max/100;//pow(0.7,num_lambda1_ss-1);
       printf(" lambda1_ss_min is: %g\n",lambda1_ss_min);
      del_log_lambda1_ss=(log10(lambda1_ss_max)-log10(lambda1_ss_min))/(num_lambda1_ss-1);
       printf(" del_log_lambda1_ss is: %g\n",del_log_lambda1_ss);
      for(i=0;i<num_lambda1_ss;i++){
        gsl_vector_set(Lambda1_ss,i,pow(10,log10(lambda1_ss_max)-del_log_lambda1_ss*i));
      }

    printf("\n" );
    printf("Lambda1_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++)
    printf("%g ",gsl_vector_get(Lambda1_ss,i));
    printf("\n");



        for(i=0;i<num_lambda1_ss;i++){
       // ProAdmmLasso(X_lasso,X_lassotX_lasso,X_lassotY,Y,beta_lasso,gsl_vector_get(Lambda1_ss,i),1,Lip_lasso,2000,N_threads);

        FISTA(X_lasso,Y, X_lassotX_lasso, X_lassotY, 2*gsl_vector_get(Lambda1_ss,i),beta_lasso,Lip_lasso);
             printf("The lasso solution is:\n");
             for(int temp_i=0;temp_i<beta_lasso->size;temp_i++){
               printf("%g ",gsl_vector_get(beta_lasso,temp_i));
             }

        for(temp_i=0;temp_i<beta_lasso->size;temp_i++){
            gsl_vector_set(beta,temp_i,gsl_vector_get(beta_lasso,temp_i));
            gsl_vector_set(beta,temp_i+p,gsl_vector_get(beta_lasso,temp_i));
            //gsl_vector_set(beta,temp_i+2*p,gsl_vector_get(beta_lasso,temp_i));
        }

        gsl_blas_dgemv(CblasNoTrans, 1, XtX, beta, 0, XtX_beta);
        gsl_vector_memcpy(Temp,XtY); /// It's very important to recopy XtY to Temp because Temp will change after calculation
        gsl_vector_sub(Temp,XtX_beta);//XtY-XtX*beta

        for(int t=0;t<(Temp->size);t++){
        temp=gsl_vector_get(Temp,t);
        gsl_vector_set(Temp,t,fabs(temp));
        //printf("Temp[%d] is %g\n",i,gsl_vector_get(Temp,t));
        }
       // printf("gsl_vector_max(Temp):%g\n",gsl_vector_max(Temp));
      lambda2_ss_max=gsl_vector_max(Temp)*2+gsl_vector_get(Lambda1_ss,i);
      printf("lambda2_ss_max is:%g\n",lambda2_ss_max);

            gsl_matrix_set(Lambda2_ss,i,0,lambda2_ss_max);
            lambda2_ss_min=lambda2_ss_max/100;//*pow(0.8,num_lambda2_ss-1);
             for(j=0;j<num_lambda2_ss;j++){
                del_log_lambda2_ss=(log10(lambda2_ss_max)-log10(lambda2_ss_min))/(num_lambda2_ss-1);
                gsl_matrix_set(Lambda2_ss,i,j,pow(10,log10(lambda2_ss_max)-del_log_lambda2_ss*j));
             }
    }


    printf("Lambda2_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++){
        printf("Lambda2[%d]: ",i);
        for(j=0;j<num_lambda2_ss;j++){
            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
            if(j==num_lambda2_ss-1) printf("\n");
        }
    }
        /// save Lambda1_SS and Lambda2_SS
         sprintf(s, "solution_SS/Lambda1_SS.dat");
         f = fopen(s, "w");
         gsl_vector_fprintf(f, Lambda1_ss, "%g");
         fclose(f);

         sprintf(s, "solution_SS/Lambda2_SS.dat");
         f = fopen(s, "w");
         gsl_matrix_fprintf(f, Lambda2_ss, "%g");
         fclose(f);



//     /**Read the lambda1_ss and lambda2_ss from the lambda1 and lambda2 calculated by lastPaper**/
//    sprintf(s, "data/Lambda1.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss; i++) {
//            fscanf(f, "%lf", &entry);
//         gsl_vector_set(Lambda1_ss,i,entry);
//        }
//    fclose(f);
//    printf("\n" );
//    printf("Lambda1_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++)
//    printf("%g ",gsl_vector_get(Lambda1_ss,i));
//    printf("\n");
//
//    sprintf(s, "data/Lambda2.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss*num_lambda2_ss; i++) {
//            row = i % num_lambda1_ss;
//            col = floor(i/num_lambda1_ss);
//            fscanf(f, "%lf", &entry);
//         gsl_matrix_set(Lambda2_ss,row,col, entry);
//        }
//    fclose(f);

//    printf("Lambda2_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++){
//        printf("Lambda2[%d]: ",i);
//        for(j=0;j<num_lambda2_ss;j++){
//            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
//            if(j==num_lambda2_ss-1) printf("\n");
//        }
//    }


    /// initialize Lip_up_temp
    for(i=0;i<rounds;i++){
        gsl_vector_set(Lip_up_temp,i,Lip);
        gsl_vector_set(Lip_down_temp,i,Lip);
    }

    /// initialize beta
    for(i=0;i<n_up;i++){
//        gsl_vector_set(up_beta,i,1.0);
 //       gsl_vector_set(down_beta,i,1.0);
//        gsl_vector_set(up_beta,i,0.0);
//        gsl_vector_set(down_beta,i,0.0);
        gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
        gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
    }

     /// initialize z
    for(int i=0;i<(up_z->size);i++){
        gsl_vector_set(up_z,i,0);
        gsl_vector_set(down_z,i,0);
       // printf("i is: %d, ThreadId=%d\n",i,omp_get_thread_num());
    }




    /// calculate Lipschitz constant
     printf("Start to calculate Lipschitz constant \n");
/******************************************************************************************/
    for (int r=0;r<rounds;r++){
         for(i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }

          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
          for(k=1;k<=num_condit;k++){
           for(i=0;i<m1/2;i++){
                     for(j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }
                            if(k==3){
                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                            if(k==4){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2, j+(k-1)*p,gsl_matrix_get(X4,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2, j+(k-1)*p, gsl_matrix_get(X4,down_id[i]-1,j));
                        }
                    }

                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
                if(k==4){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,down_id[i]-1));
                }
           }

        }
          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);

                     Lip_temp=PowerMethod(up_Xtup_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_up_temp,r,Lip_temp);
                     Lip_temp=PowerMethod(down_Xtdown_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_down_temp,r,Lip_temp);


    }
     printf("End up calculating Lipschitz constant \n");
/******************************************************************************************/
/******************************************************************************************/

//  for(i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
//        lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
//    for(j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){
//        lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);

gsl_matrix *SS_beta=gsl_matrix_calloc(n_up*2,num_lambda1_ss*num_lambda2_ss*rounds);
gsl_matrix_set_zero(SS_beta);

//#pragma omp parallel for collapse(3) private(r,up_id,down_id,i,k,j,up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X,Lip,st_down,et_down,st_up,et_up,down_beta,down_z,up_beta,up_z,f,i_lambda1,j_lambda2,lambda1_ss,lambda2_ss) shared(SS_beta,n_up,num_condit,rounds,m1,SS_index,p,X1,X2,X3,Y1,Y2,Y3,Lip_up_temp,Lip_down_temp,Lip_DtD,D,Dt,DtD,Lambda1_ss,Lambda2_ss)

  for( i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
    for( j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){

//    for( i_lambda1=0;i_lambda1<1;i_lambda1++){
//    for( j_lambda2=0;j_lambda2<1;j_lambda2++){
         //for (r=0;r<rounds;r++){
         for (int r=0;r<1;r++){
         //
         //
         gsl_matrix *ss_beta=gsl_matrix_calloc(2,n_up);
         gsl_matrix_set_zero(ss_beta);

           // printf("The i_lambda1 is:%d, j_lambda2 is:%d, r is :%d, Thread_ID:%d\n",i_lambda1,j_lambda2,r,omp_get_thread_num());
            double lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
            double lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);
            double Lip=0.0;


            /// up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X are private variables, we need to redefine them in each thread
            gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
            gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *up_Y=gsl_vector_calloc(m_up);

            gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
            gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *down_Y=gsl_vector_calloc(m_down);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector *up_z= gsl_vector_calloc(m_D);
            gsl_vector *up_beta= gsl_vector_calloc(n_up);
            gsl_vector *down_z= gsl_vector_calloc(m_D);
            gsl_vector *down_beta= gsl_vector_calloc(n_down);

            gsl_matrix_set_zero(up_X);
            gsl_matrix_set_zero(down_X);
            gsl_matrix_set_zero(up_Xtup_X);
            gsl_matrix_set_zero(down_Xtdown_X);
            gsl_vector_set_zero(up_Y);
            gsl_vector_set_zero(down_Y);

            /// initialize beta with beta_lasso
            /*for(int i=0;i<n_up;i++){
                gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
                gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
            }*/

            /// initialize beta with zero
            gsl_vector_set_zero(up_beta);
            gsl_vector_set_zero(down_beta);
            /// initialize z
            for(int i=0;i<(up_z->size);i++){
                gsl_vector_set(up_z,i,0);
                gsl_vector_set(down_z,i,0);
            }


            printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);
            //split data into two parts: the up part and down part
           // printf("Test_id is\n");
          /// assign the data into two parts
            for(int i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }
printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
           for(int k=1;k<=num_condit;k++){
            for(int i=0;i<m1/2;i++){
                 for(int j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }

                            if(k==3){
                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                        if(k==4){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2, j+(k-1)*p,gsl_matrix_get(X4,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2, j+(k-1)*p, gsl_matrix_get(X4,down_id[i]-1,j));
                        }
                    }
    printf("i[%d],j[%d]\n",i,j);
                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
                    if(k==4){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,down_id[i]-1));
                }

           }

        }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);



            /********* Main AIMNet_solver loop **********/
               /**up part**/
                Lip=gsl_vector_get(Lip_up_temp,r)+Lip_DtD;
                printf("Lip_up is %lf\n",Lip);
                //st_up=omp_get_wtime();
              /********* Main ADMM solver loop **********/
              AIMNet_Solver(N_threads,up_X,up_Y,up_beta,up_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                    //et_up=omp_get_wtime()-st_up;



               /**down part**/
                 Lip=gsl_vector_get(Lip_down_temp,r)+Lip_DtD;
                 printf("Lip_down is %lf\n",Lip);
                // st_down=omp_get_wtime();
                 AIMNet_Solver(N_threads,down_X,down_Y,down_beta,down_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                 //et_down=omp_get_wtime()-st_down;

                //printf("The total threads are %d , time_down is:%15.15lf\n",N_threads, et_down);
                //printf("The total threads are %d , time_up is: %15.15lf, time_down is:%15.15lf\n",N_threads,et_up,et_down);




           for(int j=0;j<down_beta->size;j++){
               gsl_matrix_set(ss_beta,0,j,gsl_vector_get(up_beta,j));
               gsl_matrix_set(ss_beta,1,j,gsl_vector_get(down_beta,j));
               gsl_matrix_set(SS_beta,j,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(up_beta,j));
               gsl_matrix_set(SS_beta,j+down_beta->size,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(down_beta,j));
            }

          /*sprintf(s, "solution_SS/ss_beta_%d_%d_r_%d.dat", i_lambda1,j_lambda2,r);
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, ss_beta, "%g");
            fclose(f);
          */
            gsl_matrix_free(up_X);
            gsl_matrix_free(up_Xtup_X);
            gsl_vector_free(up_Y);
            gsl_matrix_free(down_X);
            gsl_matrix_free(down_Xtdown_X);
            gsl_vector_free(down_Y);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector_free(up_z);
            gsl_vector_free(up_beta);
            gsl_vector_free(down_z);
            gsl_vector_free(down_beta);

         }

    }
  }


  sprintf(s, "solution_SS/SS_beta.dat");
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, SS_beta, "%g");
            fclose(f);

 /******************************************************************************************/

 printf("the maximum number of threads is: %d\n", omp_get_max_threads());
 printf("the number of threads used in parallel is: %d\n",N_threads);



 return 0;



}



/*
Solve the differential gene regulatory networks inference problem under three conditions.
Set the number of conditions in head.h file: num_condit=5.
The program uses OpenMP for parallel implementation and the GNU Scientific Library(GSL) for math.

Run the program in parallel:
    Use "export OMP_NUM_THREADS=num_threads" to tell the program the number of threads.
    For example:
        In the command line:
            export OMP_NUM_THREADS=16
        In the code:
            N_threads=omp_get_max_threads()-1;// which means the number of threads used in parallel is 15 (the total threads number-1).

Compile the code in local:
    gcc -Wall -std=c99 -I/GSL_path/include -c mmio.c -o mmio.o
    gcc -fopenmp -Wall -std=c99 -g -I/GSL_path/include -c simu_5condits.c -o simu_5condits.o
    gcc -fopenmp -Wall -std=c99 -g -L/GSL_path/lib  simu_5condits.o mmio.o -o simu_5condits -lgsl -lgslcblas -lm

    For example:
   	gcc -Wall -std=c99 -I/usr/local/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/usr/local/include -c simu_5condits.c -o simu_5condits.o
	gcc -fopenmp -Wall -std=c99 -g -L/usr/local/lib  simu_5condits.o mmio.o -o simu_5condits -lgsl -lgslcblas -lm

Run program:
    // export the threads used in parallel
    // demo_data: the example data, 5 is the number of tissues
    export OMP_NUM_THREADS=5
    ./simu_5condits demo_data 5

*/

int simu_5condits(char *argv[])
{
    int num_condit;
    long conv=strtol(argv[2],NULL,10); // the second parameter is the number of conditions
    num_condit=conv;

    FILE *f;
    char s[40];
    double entry,s_entry,Lip_temp;
   //double st_up,et_up,st_down,et_down;
    double Lip_data,Lip_DtD;
    double Lip,Lip_lasso,temp_i,temp;

    int i,j,k,m,m0,p,ncol_X,m_D,n_D;
    gsl_vector  *Lip_up_temp=gsl_vector_calloc(rounds);
    gsl_vector  *Lip_down_temp=gsl_vector_calloc(rounds);
    int temp_k=0;
    int ncol_y,nrow_y;
    int row,col;
    int s_m0,s_n0;
    int i_lambda1,j_lambda2;
    //int N_threads=1;
    int N_threads=omp_get_max_threads()-1;

    //double lambda1_ss,lambda2_ss;
    double lambda1_ss_max,lambda1_ss_min,lambda2_ss_max,lambda2_ss_min;
    double del_log_lambda1_ss,del_log_lambda2_ss;
    gsl_vector *Lambda1_ss=gsl_vector_calloc(num_lambda1_ss);
    gsl_matrix *Lambda2_ss=gsl_matrix_calloc(num_lambda1_ss,num_lambda2_ss);
    gsl_vector *M= gsl_vector_calloc(num_condit);//M stores the number of samples under each condition
    // M[0] is the first condition
    //gsl_vector *T=gsl_vector_calloc(1);
    mkdir("solution_SS",0700);

    /**Read data**/
    /* Need to read the data to know the sample sizes and the number of features */
    for(k=1;k<=num_condit;k++){
   // printf("k is %d\n", k);
    sprintf(s, "%s/A%d.dat", argv[1],k);
    // argv[0] represents the program name, argv[1] means the first parameter
    // printf("Reading %s\n", s);
    f = fopen(s, "r");
	if (f == NULL) {
		printf("Reading ERROR: %s does not exist, exiting.\n", s);
		exit(EXIT_FAILURE);
	}
    mm_read_mtx_array_size(f, &m0, &p);
    gsl_vector_set(M,k-1,m0);// different conditions maybe have different sample size
    }
    printf("M0 is %g.\n",gsl_vector_get(M,0));
    printf("M1 is %g.\n",gsl_vector_get(M,1));
    printf("M3 is %g.\n",gsl_vector_get(M,3));
    //printf("M2 is %g.\n",gsl_vector_get(M,2));
    for(k=1;k<=num_condit;k++){
       printf("The sample size in condition[%d] is %g.\n",k,gsl_vector_get(M,k-1));
    }
    printf("*****************\n");


    /** Defination of X,Y,beta **/
    /* define X and Y once we knew sample sizes and number of features */
    m=gsl_blas_dasum(M); //m is the total number of samples under multiple conditions
    ncol_X=num_condit*p;//ncol_X is the number of columns in X
    printf("the total sample size is %d.\n",m);


    if(fused_type==1){
    m_D=p*(num_condit-1);
    }
    if(fused_type==2){
    m_D=p*(num_condit*(num_condit-1)/2);
    }

    // the number of columns of D matrix
    n_D=p*num_condit;
    gsl_matrix *D = gsl_matrix_calloc(m_D, n_D);
    gsl_matrix *Dt = gsl_matrix_calloc(n_D, m_D);
    gsl_matrix *DtD = gsl_matrix_calloc(n_D, n_D);
    double *temp_a[m_D];
    double *temp_b[n_D];
    double *temp_c[n_D];
        for(i=0;i<m_D;i++){
            temp_a[i]=(double *)malloc(n_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_b[i]=(double *)malloc(m_D*sizeof(double));
        }

        for(i=0;i<n_D;i++){
            temp_c[i]=(double *)malloc(n_D*sizeof(double));
        }


    gsl_matrix *X= gsl_matrix_calloc(m, ncol_X); // X is the microarray expression matrix
	gsl_vector *Y= gsl_vector_calloc(m);// Y is the target gene
    gsl_vector *beta = gsl_vector_calloc(ncol_X);
    gsl_vector *beta_lasso= gsl_vector_calloc(p);//beta1=beta2=beta3=beta_lasso when calculate lambda2_max given a specific lambda1
	gsl_matrix *X_lasso= gsl_matrix_calloc(m, p);
    gsl_vector *XtY    = gsl_vector_calloc(ncol_X);
    gsl_matrix *XtX    = gsl_matrix_calloc(ncol_X,ncol_X);
    gsl_vector *v    = gsl_vector_calloc(ncol_X);
    gsl_vector *X_lassotY    = gsl_vector_calloc(p);
    gsl_matrix *X_lassotX_lasso  = gsl_matrix_calloc(p,p);
    ////gsl_vector *z= gsl_vector_calloc(m_D);


     gsl_vector *Temp=gsl_vector_calloc(ncol_X);
     gsl_vector *XtX_beta= gsl_vector_calloc(ncol_X);


//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
    int m1=gsl_vector_get(M,0);
    int m2=gsl_vector_get(M,1);
    int m3=gsl_vector_get(M,2);
    int m4=gsl_vector_get(M,3);
    int m5=gsl_vector_get(M,4);
    gsl_matrix *X1= gsl_matrix_calloc(m1, p);
    gsl_matrix *X2= gsl_matrix_calloc(m2, p);
    gsl_matrix *X3= gsl_matrix_calloc(m3, p);
    gsl_matrix *X4= gsl_matrix_calloc(m4, p);
    gsl_matrix *X5= gsl_matrix_calloc(m5, p);

    gsl_vector *Y1= gsl_vector_calloc(m1);
    gsl_vector *Y2= gsl_vector_calloc(m2);
    gsl_vector *Y3= gsl_vector_calloc(m3);
    gsl_vector *Y4= gsl_vector_calloc(m4);
    gsl_vector *Y5= gsl_vector_calloc(m5);
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//



      /** Build D matrix **/
  /************************************Build D matrix Start*****************************************************/
                    /**Build D matrix according to num_condit, lambda1 and lambda2**/

                    Build_D_matrix(D,p,num_condit,fused_type,1,1);

                    for(i=0;i<n_D;i++){
                    for(j=0;j<m_D;j++){
                    // Dt: D transpose
                    gsl_matrix_set(Dt,i,j,gsl_matrix_get(D,j,i));}
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_a,D)
                    {
                       //#pragma omp for
                    for(i=0;i<m_D;i++){
                        for(j=0;j<n_D;j++){
                            temp_a[i][j]=gsl_matrix_get(D,i,j);}
                    }
                    }

                     //#pragma omp parallel num_threads(N_threads) private(i,j) shared(temp_b,Dt)
                    {
                   // #pragma omp for
                    for(i=0;i<n_D;i++){
                        for(j=0;j<m_D;j++){
                            temp_b[i][j]=gsl_matrix_get(Dt,i,j);
                        }
                    }
                    }


                    for(i=0;i<n_D;i++){
                        for(j=0;j<n_D;j++){
                             temp_c[i][j]=0;
                             gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                            }
                     }
                  //#pragma omp parallel for schedule(static,m/N_threads) private(i,j,temp_k) shared(DtD,temp_c)
                    for(i=0;i<n_D;i++){
                        for( j=0;j<n_D;j++){
                            for(temp_k=0;temp_k<m_D;temp_k++){
                                temp_c[i][j]+=temp_b[i][temp_k]*temp_a[temp_k][j];
                            }
                            gsl_matrix_set(DtD,i,j,temp_c[i][j]);
                        }
                    }
    /************************************Build D matrix End*****************************************************/
    /** calculate Lip_DtD **/
    Lip_DtD=PowerMethod(DtD,1000,5e-3,N_threads);
    printf("Lip_DtD is %lf\n",Lip_DtD);



    /*****define variables used in the stability selection******/
    int up_id[m1/2];//firstly only consider m1=m2=m3
    int down_id[m1/2];
    int m_up,n_up,m_down,n_down;
    m_up=m/2;
    n_up=p*num_condit;
    m_down=m_up;
    n_down=n_up;
    gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
    gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
    gsl_vector *up_Y=gsl_vector_calloc(m_up);

     gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
     gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
     gsl_vector *down_Y=gsl_vector_calloc(m_down);

    //// up_z,up_x,down_z,down_x are the variables during each round ////
    gsl_vector *up_z= gsl_vector_calloc(m_D);
    gsl_vector *up_beta= gsl_vector_calloc(n_up);
    gsl_vector *down_z= gsl_vector_calloc(m_D);
    gsl_vector *down_beta= gsl_vector_calloc(n_down);


    /* initialize up part and down part in stability selection */
	for(i=0;i<m_up;i++)
    for(j=0;j<n_up;j++){
        gsl_matrix_set(up_X,i,j,0);
        gsl_matrix_set(down_X,i,j,0);
    }
    for(i=0;i<m_up;i++){
        gsl_vector_set(up_Y,i,0);
        gsl_vector_set(down_Y,i,0);
    }

    //// read the index of subsamples for each round /////
      gsl_matrix *SS_index=gsl_matrix_calloc(2*rounds,m1/2);
      sprintf(s, "SS_index_%d_.dat",m1);
    // SS_index the first row is the index for up part in the file
    f = fopen(s, "r");
    if (f == NULL) {
        printf("Reading ERROR: %s does not exist, exiting.\n", s);
        exit(EXIT_FAILURE);
    }
    mm_read_mtx_array_size(f, &s_m0, &s_n0);
    for (int i = 0; i < s_m0*s_n0; i++) {
        row = i % s_m0;
        col = floor(i/s_m0);
        fscanf(f, "%lf", &s_entry);
        gsl_matrix_set(SS_index, row, col, s_entry);
    }

//    printf("SS_index is:\n" );
//    for(i=0;i<s_m0;i++){
//        printf("SS_index[%d]: ",i);
//        for(j=0;j<s_n0;j++){
//            printf("%g ",gsl_matrix_get(SS_index,i,j));
//            if(j==s_n0-1) printf("\n");
//        }
//    }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
	/** Read data **/
    /* read X */ //the data is stored by column
    for(k=1;k<=num_condit;k++){
        sprintf(s, "%s/A%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &m0, &p);
        for (int i = 0; i < m0*p; i++) {
            row = i % m0;
            col = floor(i/m0);
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_matrix_set(X1,row,col, entry);
                gsl_matrix_set(X,row,col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso, row, col, entry);
            }
            if(k==2){
                gsl_matrix_set(X2,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0), col, entry);
            }
            if(k==3){
                gsl_matrix_set(X3,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1), col, entry);
            }
              if(k==4){
                gsl_matrix_set(X4,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2), col, entry);
            }
            if(k==5){
                gsl_matrix_set(X5,row,col, entry);
                gsl_matrix_set(X,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2)+gsl_vector_get(M,3),col+(k-1)*p, entry);
                gsl_matrix_set(X_lasso,row+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2)+gsl_vector_get(M,3), col, entry);
            }

        }
        fclose(f);

        /* Read Y */
	  sprintf(s, "%s/b%d.dat",argv[1],k);
        f = fopen(s, "r");
        if (f == NULL) {
            printf("Reading ERROR: %s does not exist, exiting.\n", s);
            exit(EXIT_FAILURE);
        }
        mm_read_mtx_array_size(f, &nrow_y,&ncol_y);
        for (int i = 0; i < nrow_y; i++) {
            fscanf(f, "%lf", &entry);
            if(k==1){
                gsl_vector_set(Y1, i, entry);
                gsl_vector_set(Y, i, entry);}
            if(k==2){
                gsl_vector_set(Y2, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0), entry);}
            if(k==3){
                gsl_vector_set(Y3, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1), entry);}
            if(k==4){
                gsl_vector_set(Y4, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2), entry);}
            if(k==4){
                gsl_vector_set(Y5, i, entry);
                gsl_vector_set(Y, i+gsl_vector_get(M,0)+gsl_vector_get(M,1)+gsl_vector_get(M,2)+gsl_vector_get(M,3), entry);}
        }
        fclose(f);

    }
//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//



            /* Precompute and cache factorizations */
     gsl_blas_dgemv(CblasTrans, 1, X, Y, 0, XtY); // XtY = X^T*Y
     gsl_vector_memcpy(Temp,XtY);
     gsl_blas_dgemv(CblasTrans, 1, X_lasso, Y, 0, X_lassotY); // XtY = X^T*Y
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X,0.0, XtX);
     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X_lasso, X_lasso,0.0, X_lassotX_lasso);
//     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)/m;
     Lip_data=PowerMethod(XtX,1000,5e-3,N_threads)*2;
     Lip=Lip_data+Lip_DtD;
     printf("Lip is %lf\n",Lip);
     Lip_lasso=PowerMethod(X_lassotX_lasso,1000,5e-3,N_threads)*2+1;



     /**calculate lambda1_ss and lambda2_ss**/
     /**lambda1_max=max(X^TY)**/
        for(int i=0;i<ncol_X;i++){
       gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
              gsl_vector_set(v,i,fabs(gsl_vector_get(XtY, i)));
                  }
       lambda1_ss_max=gsl_vector_max(v)*2;
     printf(" lambda1_ss_max is: %g\n",lambda1_ss_max);
    /* generate lambda1 */
      lambda1_ss_min=lambda1_ss_max/100;//pow(0.7,num_lambda1_ss-1);
       printf(" lambda1_ss_min is: %g\n",lambda1_ss_min);
      del_log_lambda1_ss=(log10(lambda1_ss_max)-log10(lambda1_ss_min))/(num_lambda1_ss-1);
       printf(" del_log_lambda1_ss is: %g\n",del_log_lambda1_ss);
      for(i=0;i<num_lambda1_ss;i++){
        gsl_vector_set(Lambda1_ss,i,pow(10,log10(lambda1_ss_max)-del_log_lambda1_ss*i));
      }

    printf("\n" );
    printf("Lambda1_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++)
    printf("%g ",gsl_vector_get(Lambda1_ss,i));
    printf("\n");



        for(i=0;i<num_lambda1_ss;i++){
       // ProAdmmLasso(X_lasso,X_lassotX_lasso,X_lassotY,Y,beta_lasso,gsl_vector_get(Lambda1_ss,i),1,Lip_lasso,2000,N_threads);

        FISTA(X_lasso,Y, X_lassotX_lasso, X_lassotY, 2*gsl_vector_get(Lambda1_ss,i),beta_lasso,Lip_lasso);
             printf("The lasso solution is:\n");
             for(int temp_i=0;temp_i<beta_lasso->size;temp_i++){
               printf("%g ",gsl_vector_get(beta_lasso,temp_i));
             }

        for(temp_i=0;temp_i<beta_lasso->size;temp_i++){
            gsl_vector_set(beta,temp_i,gsl_vector_get(beta_lasso,temp_i));
            gsl_vector_set(beta,temp_i+p,gsl_vector_get(beta_lasso,temp_i));
            //gsl_vector_set(beta,temp_i+2*p,gsl_vector_get(beta_lasso,temp_i));
        }

        gsl_blas_dgemv(CblasNoTrans, 1, XtX, beta, 0, XtX_beta);
        gsl_vector_memcpy(Temp,XtY); /// It's very important to recopy XtY to Temp because Temp will change after calculation
        gsl_vector_sub(Temp,XtX_beta);//XtY-XtX*beta

        for(int t=0;t<(Temp->size);t++){
        temp=gsl_vector_get(Temp,t);
        gsl_vector_set(Temp,t,fabs(temp));
        //printf("Temp[%d] is %g\n",i,gsl_vector_get(Temp,t));
        }
       // printf("gsl_vector_max(Temp):%g\n",gsl_vector_max(Temp));
      lambda2_ss_max=gsl_vector_max(Temp)*2+gsl_vector_get(Lambda1_ss,i);
      printf("lambda2_ss_max is:%g\n",lambda2_ss_max);

            gsl_matrix_set(Lambda2_ss,i,0,lambda2_ss_max);
            lambda2_ss_min=lambda2_ss_max/100;//*pow(0.8,num_lambda2_ss-1);
             for(j=0;j<num_lambda2_ss;j++){
                del_log_lambda2_ss=(log10(lambda2_ss_max)-log10(lambda2_ss_min))/(num_lambda2_ss-1);
                gsl_matrix_set(Lambda2_ss,i,j,pow(10,log10(lambda2_ss_max)-del_log_lambda2_ss*j));
             }
    }


    printf("Lambda2_ss is:\n" );
    for(i=0;i<num_lambda1_ss;i++){
        printf("Lambda2[%d]: ",i);
        for(j=0;j<num_lambda2_ss;j++){
            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
            if(j==num_lambda2_ss-1) printf("\n");
        }
    }
        /// save Lambda1_SS and Lambda2_SS
         sprintf(s, "solution_SS/Lambda1_SS.dat");
         f = fopen(s, "w");
         gsl_vector_fprintf(f, Lambda1_ss, "%g");
         fclose(f);

         sprintf(s, "solution_SS/Lambda2_SS.dat");
         f = fopen(s, "w");
         gsl_matrix_fprintf(f, Lambda2_ss, "%g");
         fclose(f);



//     /**Read the lambda1_ss and lambda2_ss from the lambda1 and lambda2 calculated by lastPaper**/
//    sprintf(s, "data/Lambda1.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss; i++) {
//            fscanf(f, "%lf", &entry);
//         gsl_vector_set(Lambda1_ss,i,entry);
//        }
//    fclose(f);
//    printf("\n" );
//    printf("Lambda1_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++)
//    printf("%g ",gsl_vector_get(Lambda1_ss,i));
//    printf("\n");
//
//    sprintf(s, "data/Lambda2.dat");
//    f = fopen(s, "r");
//        if (f == NULL) {
//            printf("Reading ERROR: %s does not exist, exiting.\n", s);
//            exit(EXIT_FAILURE);
//        }
//    for (int i = 0; i < num_lambda1_ss*num_lambda2_ss; i++) {
//            row = i % num_lambda1_ss;
//            col = floor(i/num_lambda1_ss);
//            fscanf(f, "%lf", &entry);
//         gsl_matrix_set(Lambda2_ss,row,col, entry);
//        }
//    fclose(f);

//    printf("Lambda2_ss is:\n" );
//    for(i=0;i<num_lambda1_ss;i++){
//        printf("Lambda2[%d]: ",i);
//        for(j=0;j<num_lambda2_ss;j++){
//            printf("%g ",gsl_matrix_get(Lambda2_ss,i,j));
//            if(j==num_lambda2_ss-1) printf("\n");
//        }
//    }


    /// initialize Lip_up_temp
    for(i=0;i<rounds;i++){
        gsl_vector_set(Lip_up_temp,i,Lip);
        gsl_vector_set(Lip_down_temp,i,Lip);
    }

    /// initialize beta
    for(i=0;i<n_up;i++){
//        gsl_vector_set(up_beta,i,1.0);
 //       gsl_vector_set(down_beta,i,1.0);
//        gsl_vector_set(up_beta,i,0.0);
//        gsl_vector_set(down_beta,i,0.0);
        gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
        gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
    }

     /// initialize z
    for(int i=0;i<(up_z->size);i++){
        gsl_vector_set(up_z,i,0);
        gsl_vector_set(down_z,i,0);
       // printf("i is: %d, ThreadId=%d\n",i,omp_get_thread_num());
    }




    /// calculate Lipschitz constant
     printf("Start to calculate Lipschitz constant \n");
/******************************************************************************************/
    for (int r=0;r<rounds;r++){
         for(i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }

          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
          for(k=1;k<=num_condit;k++){
           for(i=0;i<m1/2;i++){
                     for(j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }
                            if(k==3){
                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                            if(k==4){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2, j+(k-1)*p,gsl_matrix_get(X4,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2, j+(k-1)*p, gsl_matrix_get(X4,down_id[i]-1,j));
                        }
                            if(k==4){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2+m4/2, j+(k-1)*p,gsl_matrix_get(X5,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2+m4/2, j+(k-1)*p, gsl_matrix_get(X5,down_id[i]-1,j));
                        }
                    }

                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
                if(k==4){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,down_id[i]-1));
                }
                if(k==5){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2+m4/2, gsl_vector_get(Y5,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2+m4/2, gsl_vector_get(Y5,down_id[i]-1));
                }
           }

        }
          //*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
                     gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);

                     Lip_temp=PowerMethod(up_Xtup_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_up_temp,r,Lip_temp);
                     Lip_temp=PowerMethod(down_Xtdown_X,1000,5e-3,N_threads)*2;
                     gsl_vector_set(Lip_down_temp,r,Lip_temp);


    }
     printf("End up calculating Lipschitz constant \n");
/******************************************************************************************/
/******************************************************************************************/

//  for(i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
//        lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
//    for(j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){
//        lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);

gsl_matrix *SS_beta=gsl_matrix_calloc(n_up*2,num_lambda1_ss*num_lambda2_ss*rounds);
gsl_matrix_set_zero(SS_beta);

//#pragma omp parallel for collapse(3) private(r,up_id,down_id,i,k,j,up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X,Lip,st_down,et_down,st_up,et_up,down_beta,down_z,up_beta,up_z,f,i_lambda1,j_lambda2,lambda1_ss,lambda2_ss) shared(SS_beta,n_up,num_condit,rounds,m1,SS_index,p,X1,X2,X3,Y1,Y2,Y3,Lip_up_temp,Lip_down_temp,Lip_DtD,D,Dt,DtD,Lambda1_ss,Lambda2_ss)

  for( i_lambda1=0;i_lambda1<num_lambda1_ss;i_lambda1++){
    for( j_lambda2=0;j_lambda2<num_lambda2_ss;j_lambda2++){

//    for( i_lambda1=0;i_lambda1<1;i_lambda1++){
//    for( j_lambda2=0;j_lambda2<1;j_lambda2++){
         //for (r=0;r<rounds;r++){
         for (int r=0;r<1;r++){
         //
         //
         gsl_matrix *ss_beta=gsl_matrix_calloc(2,n_up);
         gsl_matrix_set_zero(ss_beta);

           // printf("The i_lambda1 is:%d, j_lambda2 is:%d, r is :%d, Thread_ID:%d\n",i_lambda1,j_lambda2,r,omp_get_thread_num());
            double lambda1_ss=gsl_vector_get(Lambda1_ss,i_lambda1);
            double lambda2_ss=gsl_matrix_get(Lambda2_ss,i_lambda1,j_lambda2);
            double Lip=0.0;


            /// up_X,down_X,up_Y,down_Y,up_Xtup_X,down_Xtdown_X are private variables, we need to redefine them in each thread
            gsl_matrix *up_X=gsl_matrix_calloc(m_up,n_up);
            gsl_matrix *up_Xtup_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *up_Y=gsl_vector_calloc(m_up);

            gsl_matrix *down_X=gsl_matrix_calloc(m_down,n_down);
            gsl_matrix *down_Xtdown_X=gsl_matrix_calloc(n_up,n_up);
            gsl_vector *down_Y=gsl_vector_calloc(m_down);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector *up_z= gsl_vector_calloc(m_D);
            gsl_vector *up_beta= gsl_vector_calloc(n_up);
            gsl_vector *down_z= gsl_vector_calloc(m_D);
            gsl_vector *down_beta= gsl_vector_calloc(n_down);

            gsl_matrix_set_zero(up_X);
            gsl_matrix_set_zero(down_X);
            gsl_matrix_set_zero(up_Xtup_X);
            gsl_matrix_set_zero(down_Xtdown_X);
            gsl_vector_set_zero(up_Y);
            gsl_vector_set_zero(down_Y);

            /// initialize beta with beta_lasso
            /*for(int i=0;i<n_up;i++){
                gsl_vector_set(up_beta,i,gsl_vector_get(beta,i));
                gsl_vector_set(down_beta,i,gsl_vector_get(beta,i));
            }*/

            /// initialize beta with zero
            gsl_vector_set_zero(up_beta);
            gsl_vector_set_zero(down_beta);
            /// initialize z
            for(int i=0;i<(up_z->size);i++){
                gsl_vector_set(up_z,i,0);
                gsl_vector_set(down_z,i,0);
            }


            printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);
            //split data into two parts: the up part and down part
           // printf("Test_id is\n");
          /// assign the data into two parts
            for(int i=0;i<m1/2;i++){
                up_id[i]=gsl_matrix_get(SS_index,2*r,i);
                down_id[i]=gsl_matrix_get(SS_index,2*r+1,i);
            }
printf("lambda1[%d]:%lf,lambda2[%d]:%lf,rounds[%d]\n",i_lambda1,lambda1_ss,j_lambda2,lambda2_ss,r);

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//
           for(int k=1;k<=num_condit;k++){
            for(int i=0;i<m1/2;i++){
                 for(int j=0;j<p;j++){
                           if(k==1){
                            gsl_matrix_set(up_X, i, j, gsl_matrix_get(X1,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i, j, gsl_matrix_get(X1,down_id[i]-1,j));
                        }
                            if(k==2){
                            gsl_matrix_set(up_X,i+m1/2, j+p, gsl_matrix_get(X2,up_id[i]-1,j));
                            gsl_matrix_set(down_X,i+m1/2, j+p, gsl_matrix_get(X2,down_id[i]-1,j));
                        }

                            if(k==3){
                            gsl_matrix_set(up_X,i+m1/2+m2/2, j+2*p,gsl_matrix_get(X3,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2, j+2*p, gsl_matrix_get(X3,down_id[i]-1,j));
                        }
                        if(k==4){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2, j+(k-1)*p,gsl_matrix_get(X4,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2, j+(k-1)*p, gsl_matrix_get(X4,down_id[i]-1,j));
                        }
                        if(k==5){
                            gsl_matrix_set(up_X,i+m1/2+m2/2+m3/2+m4/2, j+(k-1)*p,gsl_matrix_get(X5,up_id[i]-1,j));
                            gsl_matrix_set(down_X, i+m1/2+m2/2+m3/2+m4/2, j+(k-1)*p, gsl_matrix_get(X5,down_id[i]-1,j));
                        }
                    }
    printf("i[%d],j[%d]\n",i,j);
                if(k==1){
                    gsl_vector_set(up_Y, i, gsl_vector_get(Y1,up_id[i]-1));
                    gsl_vector_set(down_Y, i, gsl_vector_get(Y1,down_id[i]-1));
                }
                  if(k==2){
                    gsl_vector_set(up_Y, i+m1/2, gsl_vector_get(Y2,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2, gsl_vector_get(Y2,down_id[i]-1));
                }
                  if(k==3){
                    gsl_vector_set(up_Y, i+m1/2+m2/2, gsl_vector_get(Y3,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2, gsl_vector_get(Y3,down_id[i]-1));
                }
                    if(k==4){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2, gsl_vector_get(Y4,down_id[i]-1));
                }
                if(k==5){
                    gsl_vector_set(up_Y, i+m1/2+m2/2+m3/2+m4/2, gsl_vector_get(Y5,up_id[i]-1));
                    gsl_vector_set(down_Y, i+m1/2+m2/2+m3/2+m4/2, gsl_vector_get(Y5,down_id[i]-1));
                }

           }

        }

//*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Modify it according to the number of conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*//

             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, up_X,up_X,0.0, up_Xtup_X);
             gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, down_X,down_X,0.0, down_Xtdown_X);



            /********* Main AIMNet_solver loop **********/
               /**up part**/
                Lip=gsl_vector_get(Lip_up_temp,r)+Lip_DtD;
                printf("Lip_up is %lf\n",Lip);
                //st_up=omp_get_wtime();
              /********* Main ADMM solver loop **********/
              AIMNet_Solver(N_threads,up_X,up_Y,up_beta,up_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                   // et_up=omp_get_wtime()-st_up;



               /**down part**/
                 Lip=gsl_vector_get(Lip_down_temp,r)+Lip_DtD;
                 printf("Lip_down is %lf\n",Lip);
                 //st_down=omp_get_wtime();
                 AIMNet_Solver(N_threads,down_X,down_Y,down_beta,down_z,D,Dt,DtD,1,lambda1_ss,lambda2_ss,MAX_ITER,Lip);
                 //et_down=omp_get_wtime()-st_down;

                //printf("The total threads are %d , time_down is:%15.15lf\n",N_threads, et_down);
                //printf("The total threads are %d , time_up is: %15.15lf, time_down is:%15.15lf\n",N_threads,et_up,et_down);




           for(int j=0;j<down_beta->size;j++){
               gsl_matrix_set(ss_beta,0,j,gsl_vector_get(up_beta,j));
               gsl_matrix_set(ss_beta,1,j,gsl_vector_get(down_beta,j));
               gsl_matrix_set(SS_beta,j,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(up_beta,j));
               gsl_matrix_set(SS_beta,j+down_beta->size,i_lambda1*(num_lambda2_ss)*(rounds)+j_lambda2*(rounds)+r,gsl_vector_get(down_beta,j));
            }

          /*sprintf(s, "solution_SS/ss_beta_%d_%d_r_%d.dat", i_lambda1,j_lambda2,r);
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, ss_beta, "%g");
            fclose(f);
          */
            gsl_matrix_free(up_X);
            gsl_matrix_free(up_Xtup_X);
            gsl_vector_free(up_Y);
            gsl_matrix_free(down_X);
            gsl_matrix_free(down_Xtdown_X);
            gsl_vector_free(down_Y);

            //// up_z,up_x,down_z,down_x are the variables during each round ////
            gsl_vector_free(up_z);
            gsl_vector_free(up_beta);
            gsl_vector_free(down_z);
            gsl_vector_free(down_beta);

         }

    }
  }


  sprintf(s, "solution_SS/SS_beta.dat");
            f = fopen(s, "w");
            gsl_matrix_fprintf(f, SS_beta, "%g");
            fclose(f);

 /******************************************************************************************/

 printf("the maximum number of threads is: %d\n", omp_get_max_threads());
 printf("the number of threads used in parallel is: %d\n",N_threads);



 return 0;



}
















#endif // HEAD_TEST_H




