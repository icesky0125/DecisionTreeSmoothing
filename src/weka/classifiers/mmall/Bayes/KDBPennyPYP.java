package weka.classifiers.mmall.Bayes;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class KDBPennyPYP extends AbstractClassifier implements OptionHandler {
	private static final long serialVersionUID = 691858787988950382L;

	public static int nInstances;
	public static Instances m_Instances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt; // num of att values of each attribute
	xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTreePYP dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D

	public int m_KDB = 1; // -K
	public static double N0 = 5;
	private static double laplacePseudocount = 1;
	public static int m_Flag = 0; // 0: Laplace, 1: TAN, 2: Dirichlet

	@Override
	public void buildClassifier(Instances Instances) throws Exception {
		
		System.out.println("Using SAME_PARENT as Tying Strategy");

		m_Instances = Instances;

		// can classifier handle the data?
		getCapabilities().testWithFail(m_Instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(m_Instances);
			m_Instances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances.deleteWithMissingClass();

		// num of attributes, classes, instances
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();
		nInstances = m_Instances.numInstances();

		// paramsPerAtt[i]: the num of ith attribute values
		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i; // initialize m_Order[i]=i
		}

		double[] mi = null; // mutual information
		double[][] cmi = null; // conditional mutual information

		/*
		 * Initialize KDB structure
		 */

		/**
		 * compute mutual information and conditional mutual information
		 * respectively
		 */

		xxyDist_ = new xxyDist(m_Instances);
		xxyDist_.addToCount(m_Instances);

		mi = new double[nAttributes];
		cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);// I(Xi;C)
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);// I(Xi;Xj|C),i not
															// equal to J
		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nK = Math.min(u, m_KDB);

			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];// cmiOrder[0] is the
															// maximum value
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				m_ParentsTemp[m_Order[u]] = new int[m_Parents[u].length];
				for (int j = 0; j < m_Parents[u].length; j++) {
					m_ParentsTemp[m_Order[u]][j] = m_Parents[u][j];
				}
			}
		}
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i;
		}

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		// Print the structure
		// System.out.println("\nm_Order: " + Arrays.toString(m_Order));
		// for (int i = 0; i < nAttributes; i++) {
		// System.out.print(i + " : ");
		// if (m_Parents[i] != null) {
		// for (int j = 0; j < m_Parents[i].length; j++) {
		// System.out.print(m_Parents[i][j] + ",");
		// }
		// }
		// System.out.println();
		// }

		
//		dParameters_ = new wdBayesParametersTreePYP(m_Instances, paramsPerAtt, m_Order, m_Parents);

		// Update dTree_ based on parents
		for (int i = 0; i < nInstances; i++) {
			Instance instance = m_Instances.instance(i);
			dParameters_.update(instance);
		}

		// Convert counts to Probability
		xxyDist_.countsToProbs();
//		dParameters_.prepareForQuerying();

		// free up some space
		m_Instances = null;
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			//probs[c] = FastMath.log(xxyDist_.xyDist_.pp(c));// P(y)
			probs[c] = xxyDist_.xyDist_.pp(c);// P(y)
		}
//		System.out.println("prior"+Arrays.toString(probs));

		for (int u = 0; u < nAttributes; u++) {
			for (int c = 0; c < nc; c++) {
				double prob = dParameters_.query(instance,u,c);
//				System.out.println(prob);
				probs[c] += FastMath.log(prob);
			}
		}
//		System.out.println("1"+Arrays.toString(probs));
		SUtils.normalizeInLogDomain(probs);
//		System.out.println("2"+Arrays.toString(probs));
		SUtils.exp(probs);
//		System.out.println("3"+Arrays.toString(probs));
		return probs;
	}

//	public static void main(String[] argv) throws Exception {
//		int nExp = 1;
//		int k = 1;
//		
//		FileReader frData = new FileReader("/Users/nayyar/WData/datasets_DM/iris.arff");
//		Instances data = new Instances(frData);
//		data.setClassIndex(data.numAttributes() - 1);
//		KDBPennyPYP classifier = new KDBPennyPYP();
//		classifier.setKDB(k);
//		
//		long start = System.currentTimeMillis();
//		double errorRate = 0.0,rmse = 0.0;
//		for (int exp = 0; exp < nExp; exp++) {
//			Evaluation eval = new Evaluation(data);
//			eval.crossValidateModel(classifier, data, 2, new Random(exp));
//			errorRate+=eval.errorRate();
//			rmse+=eval.rootMeanSquaredError();
//			System.out.println("\ttmp avg error rate = "+errorRate/(exp+1));
//			System.out.println("\ttmp avg rmse = "+rmse/(exp+1));
//		}
//		errorRate/=nExp;
//		rmse/=nExp;
//		System.out.println("avg error rate = "+errorRate);
//		System.out.println("avg rmse = "+rmse);
//		System.out.println("\ntime="+(System.currentTimeMillis()-start));
//
////		KDBPenny classifier2 = new KDBPenny();
////		classifier2.setKDB(k);
////		classifier2.setFlag(0);
////		Evaluation eval2 = new Evaluation(data);
////		eval2.crossValidateModel(classifier2, data, 2, new Random(0));
////		System.out.println(eval2.toSummaryString());
//		
////		NaiveBayes nb = new NaiveBayes();
////		Evaluation eval3 = new Evaluation(data);
////		eval3.crossValidateModel(nb, data, 2, new Random(1));
////		System.out.println(eval3.toSummaryString());
//	}

	public static void main(String[] argv) {
		runClassifier(new KDBPennyPYP(), argv);
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);
		m_Discretization = Utils.getFlag('D', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public void setKDB(int k) {
		m_KDB = k;
	}

	public void setLaplacePseudocount(double m) {
		laplacePseudocount = m;
	}

	public double getLaplacePseudocount() {
		return laplacePseudocount;
	}

	public static double laplace(double freq1, double freq2, double numValues) {
		double mEsti = (freq1 + laplacePseudocount) / (freq2 + laplacePseudocount * numValues);
		return mEsti;
	}

	public void setFlag(int flag) {
		m_Flag = flag;
	}

	public int getFlag() {
		return m_Flag;
	}
	
	public void setN0(double a) {
		N0 = a;
	}
}
