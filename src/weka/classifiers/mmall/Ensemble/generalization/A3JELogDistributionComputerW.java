package weka.classifiers.mmall.Ensemble.generalization;

import weka.core.Instance;

public class A3JELogDistributionComputerW extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A3JELogDistributionComputerW(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A3JELogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
		
		//double w = (double) params.getNAttributes()/3.0 * 1.0/SUtils.NC3(params.getNAttributes());
		double w = 1;
		
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 2; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 1; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					for (int att3 = 0; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						long index1 = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
						int index2 = params.getAttributeIndexIndex(att1,att2,att3);
						
						probsClass += (params.getProbAtFullIndex(index1) * params.getParameterAtFullIndex(index2));							
					}
				}
			}
			probs[c] += (w * probsClass);
		}
	}

}
