package weka.classifiers.mmall.Ensemble.logDistributionComputation.W;

import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParameters;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A5JELogDistributionComputerW extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A5JELogDistributionComputerW(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A5JELogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
		
		//double w = (double) params.getNAttributes()/5.0 * 1.0/SUtils.NC5(params.getNAttributes());
		double w = 1;
		
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 4; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 3; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					for (int att3 = 2; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						for (int att4 = 1; att4 < att3; att4++) {
							int att4val = (int) inst.value(att4);

							for (int att5 = 0; att5 < att4; att5++) {
								int att5val = (int) inst.value(att5);

								long index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
								probsClass += (params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index));
							}
						}
					}
				}
			}
			probs[c] += (w * probsClass);
		}

	}

}
