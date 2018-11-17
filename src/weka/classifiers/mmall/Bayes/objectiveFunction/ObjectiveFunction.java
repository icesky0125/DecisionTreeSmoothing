package weka.classifiers.mmall.Bayes.objectiveFunction;

import weka.classifiers.mmall.Bayes.wdBayes;

//import lbfgsb.DifferentiableFunction;
//import lbfgsb.FunctionValues;

import weka.classifiers.mmall.optimize.DifferentiableFunction;
import weka.classifiers.mmall.optimize.FunctionValues;

public abstract class ObjectiveFunction implements DifferentiableFunction {

	protected final wdBayes algorithm;
	
	public ObjectiveFunction(wdBayes algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}