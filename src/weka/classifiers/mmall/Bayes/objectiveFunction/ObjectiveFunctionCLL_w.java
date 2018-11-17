package weka.classifiers.mmall.Bayes.objectiveFunction;

//import lbfgsb.FunctionValues;
import weka.classifiers.mmall.optimize.FunctionValues;

import weka.classifiers.mmall.Bayes.wdBayes;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionCLL_w extends ObjectiveFunction {

	public ObjectiveFunctionCLL_w(wdBayes algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		int nc = algorithm.getNc();

		// Copy params into XYParameters
		algorithm.dParameters_.copyParameters(params);

		double g[] = new double[algorithm.dParameters_.getNp()];

		int N = algorithm.getNInstances();
		int n = algorithm.getnAttributes();
		//int nc = algorithm.getNc();
		double[] myProbs = new double[nc];

		wdBayesParametersTree dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		wdBayesNode[] myNodes = new wdBayesNode[n];

		int[] order = algorithm.getM_Order();
		double mLogNC = -Math.log(nc);

		boolean m_Regularization = algorithm.getRegularization();
		boolean m_TowardsNaiveBayes = algorithm.getTowardsNaiveBayes();
		double m_Lambda = algorithm.getLambda();

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);

			// unboxed logDistributionForInstance_w
			for (int c = 0; c < nc; c++) {
				//System.out.println(xyDist.pp(c)  + " , " + dParameters.getClassParameter(c) + " = " + xyDist.pp(c) * dParameters.getClassParameter(c));
				myProbs[c] = dParameters.getClassCounts()[c] * dParameters.getClassParameter(c);
			}

			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < myNodes.length; u++) {
					wdBayesNode bNode = myNodes[u];
					//System.out.println(bNode.getXYParameter((int) instance.value(order[u]), c) + " , " +  bNode.getXYCount((int) instance.value(order[u]), c) + " = " + bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c));
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}

			SUtils.normalizeInLogDomain(myProbs);
			negLogLikelihood += (mLogNC - myProbs[x_C]);
			SUtils.exp(myProbs);

			// unboxed logGradientForInstance_w
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					if (m_TowardsNaiveBayes) {
						double reg = dParameters.getClassParameter(c) - dParameters.getClassCounts()[c];
						negLogLikelihood += m_Lambda/2 * Math.pow(reg, 2); 
						g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getClassCounts()[c]  + m_Lambda * reg;
					} else {
						negLogLikelihood += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
						g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getClassCounts()[c]  + m_Lambda * dParameters.getClassParameter(c);
					}
				} else {
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * dParameters.getClassCounts()[c];
				}
			}

			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int index = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double probability = bayesNode.getXYCount((int) instance.value(order[u]), c);
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);

					if (m_Regularization) {
						if (m_TowardsNaiveBayes) {
							double reg = parameter - probability;
							negLogLikelihood += m_Lambda/2 * reg * reg;
							g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability + m_Lambda * reg;
						} else {
							negLogLikelihood += m_Lambda/2 * parameter * parameter;
							g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability + m_Lambda * parameter;
						}
					} else {
						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability;
					}
				}
			}
		}

		FunctionValues fv = new FunctionValues(negLogLikelihood, g);
		//System.out.println("fv="+fv.functionValue+"\t"+negLogLikelihood);
		//System.out.print(negLogLikelihood);
		return fv;
	}

}