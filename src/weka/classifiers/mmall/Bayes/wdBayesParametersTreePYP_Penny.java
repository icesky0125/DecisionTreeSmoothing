package weka.classifiers.mmall.Bayes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import hdp.ProbabilityTree;
import hdp.logStirling.LogStirlingGenerator;
import weka.core.Instance;
import weka.core.Instances;

public final class wdBayesParametersTreePYP_Penny {

	private static Instances instances;
	private int n; // num of attributes
	private int nc; // num of classes

	private int[] m_ParamsPerAtt;	
	ArrayList<int[]> parentOrderforEachAtt;
	HashMap<ArrayList<Integer>, ProbabilityTree> map;
	
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
	
	public wdBayesParametersTreePYP_Penny(Instances data, int[] paramsPerAtt, ArrayList<int[]> parents) {
		instances = data;
		this.n = instances.numAttributes() - 1;
		this.nc = instances.numClasses();

		map = new HashMap<ArrayList<Integer>, ProbabilityTree>();
		m_ParamsPerAtt = paramsPerAtt;// num of values of each attributes
		parentOrderforEachAtt = parents;
	}
	

//	public void update(Instance instance) {
//		for (int u = 0; u < n; u++) {
//			for (int i = 0; i < roots[u].length; i++) {	
//				
//				int[] parents = roots[u][i].getParentOrder();
//				int nParents = parents.length;
//				int[] datapoint = new int[nParents + 1];
//				datapoint[0] = (int) instance.value(u);
//
//				for (int p = 0; p < nParents; p++) {
//					datapoint[1 + p] = (int) instance.value(parents[p]);
//				}
//				roots[u][i].addObservation(datapoint);
//			}
//		}
//	}

//	public void prepareForQuerying() {
//		ProbabilityTree tree;
//		for (int u = 0; u < roots.length; u++) {
////			System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^^u=="+u);
//			for (int i = 0; i < roots[u].length; i++) {
//				tree = roots[u][i];
//				tree.setLogStirlingCache(m_lgcache);
//				tree.smooth();
//			}
//		}
//	}

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

//	public double query(Instance instance, int u, int classValue) {
//
//		double sum = 0;
//		ProbabilityTree tree;
//		int[] datapoint;
//		int targetNodeValue;
//		int[] parents;
//		int treeCount = 0;
//		for (int i = 0; i < roots[u].length; i++) {
//			tree = roots[u][i];
//			parents = roots[u][i].getParentOrder();
//			datapoint = new int[parents.length];
//			datapoint[0] = classValue;
//			for (int p = 1; p < parents.length; p++) {
//				datapoint[p] = (int) instance.value( parents[p]);
//			}
////			 System.out.println("Querying following tree\n" +tree.printFinalPks());
////			 System.out.println("with datapoint: " + Arrays.toString(datapoint));
//			targetNodeValue = (int) instance.value(u);
//			double res = tree.query(datapoint)[targetNodeValue];
//			
//		}
//		return (double) sum / treeCount; //average over all the trees
//	}

//	public void convertCountToProbs() {
//		for(int u = 0; u < roots.length; u++){
//			for(int t = 0; t < roots[u].length;t++){
//				roots[u][t].convertCountToProbs();
//			}
//		}	
//	}
}