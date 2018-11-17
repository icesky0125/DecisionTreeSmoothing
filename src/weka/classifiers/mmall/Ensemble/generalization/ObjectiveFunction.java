package weka.classifiers.mmall.Ensemble.generalization;

//import lbfgsb.DifferentiableFunction;
//import lbfgsb.FunctionValues;

import weka.classifiers.mmall.optimize.DifferentiableFunction;
import weka.classifiers.mmall.optimize.FunctionValues;


public abstract class ObjectiveFunction implements DifferentiableFunction {

	protected final wdAnJE algorithm;
	
	public ObjectiveFunction(wdAnJE algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}
