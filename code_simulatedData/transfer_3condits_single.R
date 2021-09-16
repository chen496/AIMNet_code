
nrounds<-50
num_comdit<-3
nlambda1<-10
nlambda2<-10
p<-20


  Solution_path<-paste0("solution_SS/")
  
  dir_name<-paste0("solution_gene")
  x<-matrix(0,num_comdit*(p-1)*2,nrounds*nlambda1*nlambda2)
  dir.create(dir_name)
  for(i_lambda1 in 1:nlambda1){
    for(j_lambda2 in 1:nlambda2){
      down_x<-matrix(0,num_comdit*(p-1),nrounds)
      
      for(r in 1:nrounds){
        file_name<-paste0(Solution_path,"ss_beta_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        beta<-read.table(file_name,sep="\n",header=FALSE)
        x[,nlambda2*nrounds*(i_lambda1-1)+nrounds*(j_lambda2-1)+r]<-t(beta$V1)
        
        
      }      
      
      
    }
  }
 write.table(x,file=paste0(dir_name,"/ss_beta.csv"),sep=",",col.names = F,row.names = F) 

