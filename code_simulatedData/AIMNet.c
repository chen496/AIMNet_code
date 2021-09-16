
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
        simu_2condits(argv);
        printf("**** AIMNet Finished ****\n\n");
    }
    if(num_condit==3){
        simu_3condits(argv);
        printf("**** AIMNet Finished ****\n\n");
    }
    if(num_condit==4){
        simu_4condits(argv);
        printf("**** AIMNet Finished ****\n\n");
    }
    if(num_condit==5){
        simu_5condits(argv);
        printf("**** AIMNet Finished ****\n\n");
    }
    if(num_condit>5){
        printf("\n The number of conditions/tissues can't be greater than 5.\n");
        printf("****Please modify the code for more than 5 tissues****\n\n");
    }
}

return 0;
}
