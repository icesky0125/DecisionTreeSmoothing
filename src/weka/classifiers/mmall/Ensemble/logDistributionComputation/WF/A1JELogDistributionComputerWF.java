package weka.classifiers.mmall.Ensemble.logDistributionComputation.WF;

import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A1JELogDistributionComputerWF extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A1JELogDistributionComputerWF(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A1JELogDistributionComputerWF();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {

		probs[probs.length - 1] = 0;
		for (int c = 0; c < probs.length - 1; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index = params.getAttributeIndex(att1, att1val, c);
				probs[c] += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);
			}
		}
	}

}
