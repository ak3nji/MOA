

ACURACIA = 6
RAM = 3
CPU = 2
INSTANCIAS = 1
DESVIO = 7
KAPPAM = 12
DESVIOM = 13
KAPPAT = 10
DESVIOT = 11


folder = "cv"
algNames = c("_RSM.csv","_FS.csv","_DD_RSM.csv","_DD+FS.csv","_ARF.csv")
algRealNames = c("OSM","FS+OSM","DD+OSM","DD+FS+OSM","ARF")
datasets = c("AGR_A.arff","AGR_G.arff","covtypeNorm.arff","elecNormNew.arff","HYPER.arff","LED_A.arff","LED_G.arff", "SEA_A.arff","SEA_G.arff","SEAFD_A.arff","SEAFD_G.arff")
datasetsRealNames = c("AGR Abrupto","AGR Gradual","Forest Covertype","Electricity","HYPER","LED Abrupto","LED Gradual", "SEA Abrupto","SEA Gradual","SEAFD Abrupto","SEAFD Gradual")
makeFileName <-function(datasetName,algName){
  return (paste(datasetName,algName,sep = ""))
}

filename = makeFileName(datasets[3],algNames[3])

makePath <-function(folderString,fileName){
  return (paste(folderString,fileName,sep="/"))
}
pathString = makePath(folder,filename)

makePlot<-function(metrica,dataset){
  
  datasetName = datasets[dataset]
  datasetFileName = makeFileName(datasetName,algNames[1])
  datasetPath = makePath(folder,datasetFileName)
  #read instances row
  newDat = read.csv(datasetPath)[1]
  
  #collect data from algorithms
  for (i in 1:length(algNames)){
    datasetFileName = makeFileName(datasetName,algNames[i])
    datasetPath = makePath(folder,datasetFileName)
    auxcsv = read.csv(datasetPath)
    names(auxcsv)[metrica] <- algRealNames[i]
    newDat = data.frame(newDat,auxcsv[metrica])
  }
  
	dat = newDat
	# Create Line Chart

	# Number of Algorithms
	numAlg = length(algNames)

	# get the range for the x and y axis 
	xrange <- c(min(dat[1]), max(dat[1]))
	yrange <- c(min(dat[2:ncol(dat)]) - 10  , 100)
  c()
	# set up the plot 
	
	plot(xrange, yrange, type="n", xlab="número de instâncias", ylab="Accurácia (%)" ,main = datasetsRealNames[dataset]) 
	colors <- rainbow(numAlg) 
	linetype <- c(1:100) 
	plotchar <- seq(21,18+100,1)

	# add lines
	for (i in 1:numAlg) { 
	  lines(as.vector(t(dat[1])), as.vector(t(dat[i+1])), type="l", lwd=2, lty=linetype[1], col=colors[i], pch=plotchar[i]) 
	}

	# add a legend 

	legend(-xrange[2]/40, (yrange[2]-yrange[1])/3.4 + yrange[1], names(dat)[2:(numAlg+1)], cex=0.5, col=colors,
		pch=plotchar, lty=linetype[1], title="Algoritmos")
}

makePlot(ACURACIA,10)

