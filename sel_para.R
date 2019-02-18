library(ggplot2)
dir.input="./results_para_test/"
dir.save=dir.input
df = read.table(paste(dir.input, "parameters_cors.txt", sep=""), header=F)

# reformat the table
num = seq(1, nrow(df), by=34)
newdf = matrix(0, length(num), 28)
colnames(newdf) = c("LR", "CL", "BS", "EP", "OriMiR", "OriPr", "OriGn", "OriMe", "OriCNA", "OriComMiR", "OriComPr", "OriComGn", "OriComMe", "OriComCNA", "TestmiR", "TestPr", "TestGn", "TestMe", "TestCNA", "TestComMiR", "TestComPr", "TestComGn", "TestComMe", "TestComCNA", "AddOri", "AddOriCom", "AddTest", "AddTestCom")
for(i in 1:length(num)){
	newdf[i, ] = as.matrix(df[c(num[i]:(num[i]+3), (num[i]+5):(num[i]+9), (num[i]+11):(num[i]+15), (num[i]+17):(num[i]+21), (num[i]+23):(num[i]+27), (num[i]+29):(num[i]+30), (num[i]+32):(num[i]+33)), ])
	newdf[i, 1] = sub("lr:", "", newdf[i, 1])
	newdf[i, 2] = sub("cl:", "", newdf[i, 2])
	newdf[i, 3] = sub("bs:", "", newdf[i, 3])
	newdf[i, 4] = sub("ep:", "", newdf[i, 4])
}
head(newdf)
table(newdf[,4])

# plot the data
newdf = as.data.frame(newdf)
dat.plot <- ggplot(newdf, aes(x=LR, y=AddOriCom, shape=CL, color=BS, size=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Learning Rate") # add xlabel to plot
dat.plot <- dat.plot +  ylab("AddOriCom") # add ylabel to plot
dat.plot <- dat.plot +  ggtitle("Five Dimensional Scatterplot") 
ggsave(paste(dir.save, "addOriCom.2018-07-13.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf, aes(x=LR, y=AddTestCom, shape=CL, color=BS, size=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Learning Rate") 
dat.plot <- dat.plot +  ylab("AddTestCom") 
dat.plot <- dat.plot +  ggtitle("Five Dimensional Scatterplot") 
ggsave(paste(dir.save, "addTestCom.2018-07-13.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf, aes(x=LR, y=AddOri, shape=CL, color=BS, size=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Learning Rate") 
dat.plot <- dat.plot +  ylab("AddOri") 
dat.plot <- dat.plot +  ggtitle("Five Dimensional Scatterplot")
ggsave(paste(dir.save, "addOri.2018-07-16.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf, aes(x=LR, y=AddTest, shape=CL, color=BS, size=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Learning Rate") 
dat.plot <- dat.plot +  ylab("AddTest") 
dat.plot <- dat.plot +  ggtitle("Five Dimensional Scatterplot")
ggsave(paste(dir.save, "addTest.2018-07-16.jpg", sep=""), width = 7, height = 14, dpi=300)

newdf2 = newdf[which(newdf[,1] == 0.01), ]
dat.plot <- ggplot(newdf2, aes(x=CL, y=AddOriCom, shape=BS, color=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Corruption Level") 
dat.plot <- dat.plot +  ylab("AddOriCom") 
dat.plot <- dat.plot +  ggtitle("Four Dimensional Scatterplot")
ggsave(paste(dir.save, "addOriCom.LR0.01.2018-07-13.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf2, aes(x=CL, y=AddTestCom, shape=BS, color=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Corruption Level") 
dat.plot <- dat.plot +  ylab("AddTestCom") 
dat.plot <- dat.plot +  ggtitle("Four Dimensional Scatterplot")
ggsave(paste(dir.save, "addTestCom.LR0.01.2018-07-13.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf2, aes(x=CL, y=AddOri, shape=BS, color=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Corruption Level")
dat.plot <- dat.plot +  ylab("AddOri") 
dat.plot <- dat.plot +  ggtitle("Four Dimensional Scatterplot")
ggsave(paste(dir.save, "addOri.LR0.01.2018-07-16.jpg", sep=""), width = 7, height = 14, dpi=300)

dat.plot <- ggplot(newdf2, aes(x=CL, y=AddTest, shape=BS, color=EP))
dat.plot <- dat.plot + geom_point()
dat.plot <- dat.plot +  xlab("Corruption Level") 
dat.plot <- dat.plot +  ylab("AddTest") 
dat.plot <- dat.plot +  ggtitle("Four Dimensional Scatterplot")
ggsave(paste(dir.save, "addTest.LR0.01.2018-07-16.jpg", sep=""), width = 7, height = 14, dpi=300)

# get the overlap
top20.addOriCom = newdf2[order(newdf2$AddOriCom, decreasing=T)[1:20], c(1:4, 25:28)]
top20.addTestCom = newdf2[order(newdf2$AddTestCom, decreasing=T)[1:20], c(1:4, 25:28)]
para.top20.addTestCom = paste(top20.addTestCom$CL, top20.addTestCom$BS, top20.addTestCom$EP, sep=".")
para.top20.addOriCom = paste(top20.addOriCom$CL, top20.addOriCom$BS, top20.addOriCom$EP, sep=".")
intersect(para.top20.addOriCom, para.top20.addTestCom)

