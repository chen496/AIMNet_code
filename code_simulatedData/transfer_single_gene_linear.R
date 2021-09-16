nlambda1<-10
nlambda2<-10
nrounds<-50
p<-20
num_comdit<-2


  Solution_path<-"solution_SS/"
  
  dir_name<-"solution_gene"
  dir.create(dir_name)
  x<-matrix(0,num_comdit*(p-1)*2,nrounds*nlambda1*nlambda2)
  for(i_lambda1 in 1:nlambda1){
    for(j_lambda2 in 1:nlambda2){
      
      for(r in 1:nrounds){
        file_name<-paste0(Solution_path,"ss_beta_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        beta<-read.table(file_name,sep="\n",header=FALSE)
        x[,nlambda2*nrounds*(i_lambda1-1)+nrounds*(j_lambda2-1)+r]<-t(beta$V1)
        
      }      
      
      
    }
  }
  
 write.table(x,file=paste0(dir_name,"/ss_beta.csv"),sep=",",col.names = F,row.names = F)

