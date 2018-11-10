# Genetic Algorithm
library(caret)
library(e1071)
library(parallelSVM)

# Set Initial population for feature selection 
initPop <- function(popSize, col) {
	set.seed(123)
	return(data.frame(replicate(col, sample(c(0,1), popSize, rep=TRUE))))
}

# Compute fitness 
computeFitness <- function(train,pop) {
	train.cls <- train[,1]
	train.noclass <- train[,-1]
	aList <- c()
	fList <- c()
	fitList <- c()

	for (i in 1:nrow(pop)) {
  		idx <- as.logical(pop[i,])
  		subTrain <- cbind(train.cls, train.noclass[,idx])
  		# 2 fold cross validation in SVM
  		foldNo <- 2
  		set.seed(123)
		fold <- createFolds(subTrain[,1], k=foldNo)
		F1.list <- c()
		Acc.list <- c()
		sv.list <- c()

		for(j in 1:length(fold)){
    		svm.model <- svm(subTrain[-fold[[j]],-1], subTrain[-fold[[j]],1], kernel="linear", scale=F)
    		pred <- predict(svm.model, subTrain[fold[[j]],-1])
    		svm.table <- table(pred, subTrain[fold[[j]],1])
    		cm <- confusionMatrix(svm.table)
    		precision <- cm$byClass['Pos Pred Value']
    		recall <- cm$byClass['Sensitivity']
    		F1 <- 2*((precision*recall)/(precision+recall))
    		F1.list <- c(F1.list, F1)
    		Acc.list <- c(Acc.list, cm$overall["Accuracy"])
    		# Support vector
    		svNum <- sum(svm.model$nSV)
			sv.list <- c(sv.list, svNum)
		}
		F1.avg <- sum(F1.list)/foldNo
		Acc.avg <- sum(Acc.list)/foldNo
		# number of features
		featuresNum <- sum(pop[i,]==1)
		f <- 1-featuresNum/ncol(pop)
		# numbers of support vector
		sv.avg <- sum(sv.list)/foldNo
		v <- 1-sv.avg/(ncol(pop)*nrow(pop))
		# Based on paper https://www.hindawi.com/journals/sp/2016/2739621/
		# Fitness with F1 weight, No of features weight and support vector weigth 
		fitness <- 0.8 * Acc.avg + 0.15*f + 0.05*v
		
  		aList <- c(aList, Acc.avg)
  		fList <- c(fList, F1.avg)
  		fitList <- c(fitList, fitness)
	}
	
	return(list(fitList, sum(aList)/nrow(pop), sum(fList)/nrow(pop)))
}
# Tournament selection method
tournamentSelection <- function(fList, t) {
	selectedGroup <- c()
	for (i in 1:(length(fList)/2)) {
		# Generate 2 random variables to access tournament
		parents <- c()
		for (j in 1:2) {
			best <- 0
			for (i in 1:t) {
				# Higher fitness will have higher probability of being selected
				ind <- sample(1:length(fList),1, prob=fList)
				if (fList[best] < fList[ind] || best == 0) {
					best <- ind
				}
			}
			parents <- c(parents, best)
		}
		selectedGroup <- rbind(selectedGroup, parents)
	}
	return(selectedGroup)	
}
# Normalise individual in a population 
normPop <- function(fList) {
	#provide probs of individual
	#return((f1List - min(f1List))/(max(f1List)-min(f1List)))
	return(fList/sum(fList))
}

roulettewheelSelection <- function(rankPop) {
	# Generate the roulette wheel wuith cumulative sum
	rwValues <- cumsum(rankPop)
	selectedGroup <- c()
	for (i in 1:(length(rwValues)/2)) {
		# Generate 2 random variables to access roulette wheel
		parents <- c()
		for (j in 1:2) {
			s <- runif(1, min=0, max=1)
			ind <- min(which(rwValues >= s))
			parents <- c(parents, ind)
		}
		selectedGroup <- rbind(selectedGroup, parents)
	}
	return(selectedGroup)	
}

fitnessEval<- function(fList) {
	np <- normPop(fList)
	return(roulettewheelSelection(np))
}

# Crossover to produce 2 chromosome 
crossover <- function(pop, fitIdx) {
	
	cross <- sample(c(0,1), nrow(fitIdx), rep=TRUE, prob=c(0.4, 0.6))
	children <- c()
	for (i in 1:nrow(fitIdx)) {
		# direct copy parents
		if (cross[i] == 0) {
			child1 <- pop[fitIdx[i,1],]
			children <- rbind(children,child1)
			child2 <- pop[fitIdx[i,2],]
			children <- rbind(children,child2)
			 
		}
		# Perform crossover
		else {
			parent1 <- pop[fitIdx[i,1],]
			parent2 <- pop[fitIdx[i,2],]
			# Randomly generated crossover point
			cpoint <- sample(3:(ncol(pop)-1), 1)
			child1 <- cbind(parent1[,1:(cpoint-1)], parent2[,cpoint:ncol(parent2)])
			children <- rbind(children,child1)
			child2 <- cbind(parent2[,1:(cpoint-1)], parent1[,cpoint:ncol(parent1)])
			children <- rbind(children,child2)
			
		}
	}
	return(children)
}

mutation <- function(pop) {
	#Probability of mutation is 0.1%
	mut<- sample(c(0,1), nrow(pop) * ncol(pop), rep=TRUE, prob=c(0.999, 0.001))
	mutIdx <- matrix(mut, nrow=nrow(pop), ncol=ncol(pop))
	midx <- which(mutIdx == 1,arr.ind=TRUE)
	ifelse (pop[midx] == 0, pop[midx] <- 1, pop[midx] <- 0)
	return(pop)
}

plotGA <- function(maxf, meanf, accf, f1f, fNum) {
	par(mfrow=c(1,1))
	plot(maxf, pch="*", type="b", col="purple", ylim=c(0.7,1), xlim=c(0,50), main="Genetic Algorithm with SVM ", xlab="Generations", ylab="Fitness-score")
	lines(meanf, pch=".", type="b", col="green")
	lines(accf, pch=".", type="b", col="blue")
	lines(f1f, pch=".", type="b", col="red")
	legend("bottomright", c("Best   ", "Mean    ", "Accuracy     ", "F1-score   "), fill =c("purple", "green", "blue", "red"), xjust=0.5)
	#plot(x=fNum,y=maxf, pch="*", type="b", col="orange", ylim=c(0.7,1), xlim=c(100,300), main="Number of features vs fitness score", xlab="Number of feature", ylab="Fitness-score")
}

heading <- function() {
	print("************ Generic Algorithm *************")
	print("********************************************")
	print(paste("Population size:", popSize))
	print(paste("Number of generations:", maxGen))
	print("Fitness selection method: Tournament (size=5)")
	print("Probability of crossover: 70%")
	print("Probability of mutation: 0.01%")

	
}	

gamodel <- function(train, popSize, maxGen) {
	# Variables
	maxfitList <- c()
	avgfitList <- c()
	accList <- c()
	f1List <- c()
	fitList <- c()
	bestInd <- 0
	topChromosome <- c()
	bestChromosome <- c()
	goodChromosome <- c()
	featureList <- c()
	# Stopping criteria, no improvement in 10 generations
	maxBest <- 10

	# Generate heading
	heading()
	# Randomly generate initial population 
	pop <- initPop(popSize, ncol(train[,-1]))

	# loop thru the number of generation
	for (i in 1:maxGen)
	{
		# Evaluation
		fit <- computeFitness(train, pop)
		fitList <- as.numeric(fit[[1]])
		accList <- c(accList, unlist(fit[[2]]))
		f1List <- c(f1List, unlist(fit[[3]]))
		# Generate a max list and mean list for plotting
		maxfitList <- c(maxfitList, max(fitList))
		avgfitList <- c(avgfitList, mean(fitList))
		# Get all the max value of chromosome
		goodChromosome <- pop[which(fitList==max(fitList)),]
		# Choose the one with the minimum features
		bestChromosome <-  goodChromosome[which.min(rowSums(goodChromosome)),]
		# Choose the top chromosome
		
		featureList <- c(featureList, sum(bestChromosome))
		# Get max of fList
		if (bestInd < max(fitList)) {
			bestInd <- max(fitList)
			bestNo <- 0
			topChromosome <- bestChromosome
			
		}
		# Determine exit criteria
		if (bestInd == max(fitList)) {
			bestNo <- bestNo + 1
			if (sum(topChromosome) > sum(bestChromosome)) {
				topChromosome <- bestChromosome
			}
		}

		# Exit criteria, if fitness == 100% or no improvement in the last 10 generation
  		if (any(round(fitList,4)==1) || bestNo == maxBest) {
  			print("Exit before the end of generations")
  			break
  		}
  		else {
  			# Select individuals using roulette wheel
			#fitIndex <- fitnessEval(fitList)
			# Select individual using tournament selection
			tournament_size <- 10
			fitIndex <- tournamentSelection(fitList, tournament_size)
			# Perform crossover
			crossPop <- crossover(pop, fitIndex)
			# Perform mutation
			newPop <- mutation(crossPop) 
			pop <- newPop
			
		}
	}
	
	#print("===============================================")
	print("***************** SUMMARY *****************")
	print(paste("Best Number of features :", sum(topChromosome)))
	print(paste("Best Fitness-score :", round(bestInd,4)))
	print("********************************************")
	plotGA(maxfitList, avgfitList, accList, f1List, featureList)
	
	return(topChromosome)
}

