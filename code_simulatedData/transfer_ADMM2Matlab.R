
nrounds<-50
num_comdit<-2
nlambda1<-10
nlambda2<-10
m<-200
p<-20
NZ<-40
NC<-8
var<-0.5

for(i_gene in 1:p){
  
  Solution_path<-paste0("Yeast_size",p,"_NZ",NZ,"_NC",NC,"m",m,"var",var,"_",i_gene,"/solution_SS/")
  
  dir_name<-paste0("solution_gene",i_gene)
  dir.create(dir_name)
  for(i_lambda1 in 1:nlambda1){
    for(j_lambda2 in 1:nlambda2){
down_x<-matrix(0,num_comdit*(p-1),nrounds)
      down_z<-matrix(0,(num_comdit-1)*(p-1),nrounds)
      up_x<-matrix(0,num_comdit*(p-1),nrounds)
      up_z<-matrix(0,(num_comdit-1)*(p-1),nrounds)
      
      for(r in 1:nrounds){
        file_name<-paste0(Solution_path,"down_ss_beta_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        beta<-read.table(file_name,sep="\n",header=FALSE)
        down_x[,r]<-t(beta$V1)
        
        file_name<-paste0(Solution_path,"down_ss_z_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        z<-read.table(file_name,sep="\n",header=FALSE)
        down_z[,r]<-t(z$V1)
        
        file_name<-paste0(Solution_path,"up_ss_beta_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        beta<-read.table(file_name,sep="\n",header=FALSE)
        up_x[,r]<-t(beta$V1)
        
        file_name<-paste0(Solution_path,"up_ss_z_",i_lambda1-1,"_",j_lambda2-1,"_r_",r-1,".dat")
        z<-read.table(file_name,sep="\n",header=FALSE)
        up_z[,r]<-t(z$V1)
        
      }      
      
      write.table(down_x,file=paste0(dir_name,"/down_ss_beta_",i_lambda1,"_",j_lambda2,".csv"),sep=",",col.names = F,row.names = F)
      write.table(down_z,file=paste0(dir_name,"/down_ss_z_",i_lambda1,"_",j_lambda2,".csv"),sep=",",col.names = F,row.names = F)
      write.table(up_x,file=paste0(dir_name,"/up_ss_beta_",i_lambda1,"_",j_lambda2,".csv"),sep=",",col.names = F,row.names = F)
      write.table(up_z,file=paste0(dir_name,"/up_ss_z_",i_lambda1,"_",j_lambda2,".csv"),sep=",",col.names = F,row.names = F)
      
    }
  }
  
}
