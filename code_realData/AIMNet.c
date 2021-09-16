#include "head.h"
int main(int argc,char *argv[])
{

printf("**** AIMNet Started ****\n\n");
if(argc<=2)
{
    printf("****Please input the data path and the number of conditions****\n");
    printf("****Exit and try it again****\n");
}else{

    long num_condit=strtol(argv[2],NULL,10);

    if(num_condit==2){
        realdata_2condits_cv(argv);
        printf("**** AIMNet Cross Validation Finished: 2 condits ****\n\n");
        realdata_2condits_ss(argv);
        printf("**** AIMNet Stability Selection Finished: 2 condits ****\n\n");
    }
    if(num_condit==3){
        realdata_3condits_cv(argv);
        printf("**** AIMNet Cross Validation Finished: 3 condits ****\n\n");
        realdata_3condits_ss(argv);
        printf("**** AIMNet Stability Selection Finished: 3 condits ****\n\n");
    }
    if(num_condit==4){
        realdata_4condits_cv(argv);
        printf("**** AIMNet Cross Validation Finished: 4 condits ****\n\n");
        realdata_4condits_ss(argv);
        printf("**** AIMNet Stability Selection Finished: 4 condits ****\n\n");
    }
    if(num_condit==5){
        realdata_5condits_cv(argv);
        printf("**** AIMNet Cross Validation Finished: 5 condits ****\n\n");
        realdata_5condits_ss(argv);
        printf("**** AIMNet Stability Selection Finished: 5 condits ****\n\n");
    }
    if(num_condit>5){
        printf("\n The number of conditions/tissues can't be greater than 5.\n");
        printf("****Please modify the code for more than 5 tissues****\n\n");
    }
}
return 0;
}
