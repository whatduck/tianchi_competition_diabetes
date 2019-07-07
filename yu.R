data=read.table('C:/Users/Administrator/Desktop/消费贷款数据/消费贷款数据.csv',sep=',',header = T)
summary(data$申请人数)
data$申请人数[data$申请人数==0]=1
mydata=data
summary(mydata)


###############分词申请条件
wz=paste(mydata$申请条件,collapse = '')
summary(wz)
library(jiebaR)
cutter=worker()
segment=cutter[wz]
worfre=freq(segment)
#head(worfre)


###########提取高频词
library(wordcloud2)
#wordcloud2(worfre)
newword=subset(worfre,nchar(worfre$char)!=1)
#wordcloud2(newword)
head(newword[order(newword$freq,decreasing = T),],40)
aa=head(newword[order(newword$freq,decreasing = T),],100)
wordcloud2(aa)


#################因变量
newdata=data.frame(mydata$公司名称)
newdata$申请人数=log(mydata$申请人数)


#################自变量：延伸变量5个
newdata$年龄[grep('周岁|年龄|岁',mydata$申请条件)]=1
newdata$年龄[is.na(newdata$年龄)]=0
newdata$收入[1046]=0
newdata$收入[grep('工作|流水|工资|收入',mydata$申请条件)]=1
newdata$收入[is.na(newdata$收入)]=0
newdata$财产[grep('房产|车',mydata$申请条件)]=1
newdata$财产[is.na(newdata$财产)]=0
newdata$信用[grep('信用|征信',mydata$申请条件)]=1
newdata$信用[is.na(newdata$信用)]=0
newdata$社保[1046]=0
newdata$社保[grep('社保|公积金|保单|保险',mydata$申请条件)]=1
newdata$社保[is.na(newdata$社保)]=0


#############自变量：变量整理
#自变量：城市
summary(mydata$城市)
newdata$一线[mydata$城市%in%c("北京","上海","广州","深圳")]=1
newdata$一线[is.na(newdata$一线)]=0
newdata$二线[mydata$城市%in%c("成都","重庆","杭州","天津","西安")]=1
newdata$二线[is.na(newdata$二线)]=0


#自变量：月管理费
testdata<-data.frame(mydata$月管理费数据)
月费=lapply(testdata, function(x) as.numeric(sub("%", "", x))/100)
newdata$月费=月费$mydata.月管理费数据

#自变量：期限最低范围、期限最高范围
tapply(mydata$申请人数,mydata$期限最低范围,mean)
newdata$lowtim1[mydata$期限最低范围%in%c(1,3)]=1
newdata$lowtim1[is.na(newdata$lowtim1)]=0
newdata$lowtim2[mydata$期限最低范围==6]=1
newdata$lowtim2[mydata$期限最低范围!=6]=0

tapply(mydata$申请人数,mydata$期限最高范围,mean)
summary(mydata$期限最高范围)
summary(as.factor(mydata$期限最高范围))

newdata$hightim1[mydata$期限最高范围<=24]=1
newdata$hightim1[is.na(newdata$hightim1)]=0
newdata$hightim2[mydata$期限最高范围>24&mydata$期限最高范围<=36]=1
newdata$hightim2[is.na(newdata$hightim2)]=0

#自变量：还款方式

tapply(mydata$申请人数,mydata$还款方式,mean)
newdata$repay1[mydata$还款方式=="分期还款"]=1
newdata$repay1[mydata$还款方式!="分期还款"]=0
newdata$repay2[mydata$还款方式=="到期还款"]=1
newdata$repay2[mydata$还款方式!="到期还款"]=0

#自变量：放款日期、审批时间


tapply(mydata$申请人数,mydata$放款日期,mean)
summary(mydata$放款日期)
summary(as.factor(mydata$放款日期))


newdata$loantim1[mydata$放款日期%in%c(0:3)]=1
newdata$loantim1[is.na(newdata$loantim1)]=0
newdata$loantim2[mydata$放款日期%in%c(4:7)]=1
newdata$loantim2[is.na(newdata$loantim2)]=0

tapply(mydata$申请人数,mydata$审批时间,mean)
summary(mydata$审批时间)
summary(as.factor(mydata$审批时间))

newdata$examtim1[mydata$审批时间%in%c(0:3)]=1
newdata$examtim1[is.na(newdata$examtim1)]=0


#自变量：担保方式

tapply(mydata$申请人数,mydata$担保方式,mean)
b=as.data.frame(a)

newdata$guar1[mydata$担保方式=="信用贷"]=1
newdata$guar1[is.na(newdata$guar1)]=0
newdata$guar2[mydata$担保方式=="担保贷"]=1
newdata$guar2[is.na(newdata$guar2)]=0
newdata$guar3[mydata$担保方式=="抵押贷"]=1
newdata$guar3[is.na(newdata$guar3)]=0


#############线性回归
#多元线性
fit=lm(申请人数~.-1-mydata.公司名称,data=newdata)
summary(fit)
par(mfrow=c(2,2))
plot(fit)
#逐步
tstep<-step(fit)
summary(tstep)
par(mfrow=c(2,2))
plot(tstep)


#############描述性统计

library(ggplot2)
require(gridExtra)
#因变量
par(mfrow=c(2,2))
ggplot(data=mydata) + geom_histogram(aes(x=申请人数),bins=30,fill="lightblue")



#自变量：城市
mydata=subset(mydata,mydata$申请人数<750)
pcity1=ggplot(mydata,aes(y=申请人数, x=城市)) + geom_boxplot(fill="lightblue")
申请人数均值=tapply(mydata$申请人数,mydata$城市,mean)
b=as.data.frame(申请人数均值)
pcity2=ggplot(data = b, mapping = aes( x=row.names(b),y=申请人数均值)) + geom_bar(stat = 'identity',fill="lightblue")
grid.arrange(pcity1,pcity2, ncol=1, nrow=2)


#自变量：信贷模式

mydata1=subset(mydata,mydata$申请人数<750)
repay1=ggplot(mydata1,aes(y=申请人数, x=还款方式)) + geom_boxplot(fill="lightblue")
申请人数均值=tapply(mydata$申请人数,mydata$还款方式,mean)
b=as.data.frame(申请人数均值)
repay2=ggplot(data = b, mapping = aes( x=row.names(b),y=申请人数均值)) + geom_bar(stat = 'identity',fill="lightblue")

gur1=ggplot(mydata1,aes(y=申请人数, x=担保方式)) + geom_boxplot(fill="lightblue")
申请人数均值=tapply(mydata$申请人数,mydata$担保方式,mean)
d=as.data.frame(申请人数均值)
gur2=ggplot(data = d, mapping = aes( x=row.names(d),y=申请人数均值)) + geom_bar(stat = 'identity',fill="lightblue")

grid.arrange(repay1,repay2, gur1,gur2,ncol=2, nrow=2)




