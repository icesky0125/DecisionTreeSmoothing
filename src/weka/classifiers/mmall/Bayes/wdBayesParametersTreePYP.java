package weka.classifiers.mmall.Bayes;

import hdp.ProbabilityTree;
import hdp.logStirling.LogStirlingGenerator;
import weka.core.Instance;
import weka.core.Instances;

public final class wdBayesParametersTreePYP {

//	private ProbabilityTree[] pypTrees;

	private static Instances instances;
	private int n; // num of attributes
	private int nc; // num of classes

	private int[] m_ParamsPerAtt;
	private int[] order;
	private int[][][] parents;
	private double m_Discount;
	
	public ProbabilityTree[][] roots;
	LogStirlingGenerator m_lgcache;

	/**
	 * Constructor called by wdBayes
	 */
//	public wdBayesParametersTreePYP(Instances data, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents) {
//
//		instances = data;
//		this.n = instances.numAttributes() - 1;
//		this.nc = instances.numClasses();
//
//		m_ParamsPerAtt = new int[n]; // num of values of each attributes
//		for (int u = 0; u < n; u++) {
//			m_ParamsPerAtt[u] = paramsPerAtt[u];
//		}
//
//		order = m_Order;
////		parents = m_Parents;
//
//		
//		pypTrees = new ProbabilityTree[n];
//		for (int u = 0; u < n; u++) {
//			int nParents = (m_Parents[u] == null) ? 0 : m_Parents[u].length;
//
//			// arityConditioningVariables is values for every attribute
//			int[] arityConditioningVariables = new int[1 + nParents];// +1 for
//																		// class
//			arityConditioningVariables[0] = nc;
//			for (int p = 1; p < arityConditioningVariables.length; p++) {
//				arityConditioningVariables[p] = paramsPerAtt[parents[u][0][p - 1]];
//			}
//			// pypTrees[u] = new ProbabilityTree(paramsPerAtt[m_Order[u]],
//			// arityConditioningVariables);
//			// create full tree
//			pypTrees[u] = new ProbabilityTree(paramsPerAtt[m_Order[u]], arityConditioningVariables, true);
//		}
//	}
	
	public wdBayesParametersTreePYP(Instances data, int[] paramsPerAtt, int[] m_Order, int[][][] m_Parents, int gibbs) {
		instances = data;
		this.n = instances.numAttributes() - 1;
		this.nc = instances.numClasses();

		m_ParamsPerAtt = paramsPerAtt;// num of values of each attributes
		order = m_Order;
		parents = m_Parents;
		roots = new ProbabilityTree[n][];

		for (int u = 0; u < n; u++) {
			int[][] parentsU = parents[u];
		
			roots[u] = new ProbabilityTree[parentsU.length];
			for (int i = 0; i < roots[u].length; i++) {
				int[] parentOrderU = parentsU[i];
				int nParents = parentOrderU.length;
				
				// arityConditioningVariables is values for every parent attribute
				int[] arityConditioningVariables = new int[nParents];
				for(int p = 0; p < nParents; p++){
					arityConditioningVariables[p] = m_ParamsPerAtt[parentOrderU[p]];
				}
				
				ProbabilityTree root = new ProbabilityTree(m_ParamsPerAtt[u], arityConditioningVariables, true);
				roots[u][i] = root;
				root = null;
			}	
		}
	}
	
	public wdBayesParametersTreePYP(Instances m_Instances, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents,
			int m_Iterations, int m_Tying) {
		// TODO Auto-generated constructor stub
	}

	public void update(Instance instance) {
		
		for (int u = 0; u < roots.length; u++) {
			
			for (int i = 0; i < roots[u].length; i++) {				
				int nParents = parents[u][i].length;
				int[] datapoint = new int[nParents + 1];
				datapoint[0] = (int) instance.value(u);

				for (int p = 0; p < nParents; p++) {
					datapoint[1 + p] = (int) instance.value(parents[u][i][p]);
				}

				roots[u][i].addObservation(datapoint);
			}
		}
	}

	public void prepareForQuerying() {
		ProbabilityTree tree;
		for (int u = 0; u < roots.length; u++) {
//			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^u=="+u);
			for (int i = 0; i < roots[u].length; i++) {
				tree = roots[u][i];
				tree.setLogStirlingCache(m_lgcache);
				tree.smooth();
			}
		}
	}

//	public double[] query(Instance instance, int u) {
//		ProbabilityTree trees = pypTrees[u];
//		int nParents = (parents[u] == null) ? 0 : parents[u].length;
//		
//		// test data 
//		int[] datapoint = new int[nParents + 1];
//		datapoint[0] = (int) (int) instance.classValue();
//		for (int p = 0; p < nParents; p++) {
//			datapoint[1 + p] = (int) instance.value(parents[u][p]);
//		}
////		System.out.println("Querying following tree\n" + tree.printFinalPks());
////		System.out.println("with datapoint: " + Arrays.toString(datapoint));
//
//		return tree.query(datapoint);
//	}

	public double query(Instance instance, int u, int classValue) {

		double sum = 0;
		ProbabilityTree tree;
		int depth;
		int[] datapoint;
		int targetNodeValue;
		for (int i = 0; i < roots[u].length; i++) {
			tree = roots[u][i];
			depth = tree.getNXs();
			datapoint = new int[depth];

			datapoint[0] = classValue;
			for (int p = 1; p < depth; p++) {
				datapoint[p] = (int) instance.value(parents[u][i][p]);
			}
//			 System.out.println("Querying following tree\n" +tree.printFinalPks());
//			 System.out.println("with datapoint: " + Arrays.toString(datapoint));
			targetNodeValue = (int) instance.value(u);
			double res = tree.query(datapoint)[targetNodeValue];
//			System.out.println(res);

			sum += res;
		}

		return (double) sum / roots[u].length; //average over all the trees
	}

//	public void convertCountToProbs() {
//		for(int u = 0; u < roots.length; u++){
//			for(int t = 0; t < roots[u].length;t++){
//				roots[u][t].convertCountToProbs();
//			}
//		}	
//	}

	public void setDiscount(double d) {
		m_Discount = d;
	}
	
	public void printAllTrees(){
		for(int i = 0; i < roots.length;i++){
//			System.out.println("u == "+i);
			for(int j = 0; j < roots[i].length; j++){
				 System.out.println(roots[i][j].printFinalPks());
			}
		}
	}

	public void setLogStirlingCache(LogStirlingGenerator lgcache) {
		m_lgcache = lgcache;
	}
}