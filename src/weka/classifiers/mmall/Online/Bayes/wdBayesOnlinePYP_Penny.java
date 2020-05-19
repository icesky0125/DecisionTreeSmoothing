/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2016 Francois Petitjean, Nayyar A Zaidi and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdBayesOptMT Classifier
 * 
 * wdBayesOptMT.java     
 * Code written by:  Francois Petitjean, Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * -V 	Verbosity
 * -S	Structure learning (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis)
 * -P	Parameter learning (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN, 5: PYP)
 * -K 	Value of K for KDB.
 * 
 */
package weka.classifiers.mmall.Online.Bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import method.SmoothingMethod;
import hdp.ProbabilityTree;
import hdp.TyingStrategy;
import hdp.logStirling.LogStirlingGenerator;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public final class wdBayesOnlinePYP_Penny {

	private Instances m_Instances;
	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	private Instances structure;

	private xxyDist xxyDist_;
//	public wdBayesParametersTreePYP dParameters_;
	private String m_S = "NB";
	private int m_KDB = 1; // -K
	private boolean m_MVerb = false; // -V
	private RandomGenerator rg = null;
	private BNStructure bn = null;
	private static final int BUFFER_SIZE = 100000;
//	int m_BestattIt = 0;
	private static int m_IterGibbs = 50000;
	int m_Tying = 2;
	
	// added by He Zhang
	private double m_Discount;
	ArrayList<ArrayList<ArrayList<Integer>>> parentOrderforEachAtt;
	private int ensembleSize = 1;
	ArrayList<HashMap<ArrayList<Integer>, ProbabilityTree>> map;
	ArrayList<int[][][]> parentOrder;
	private ArrayList<int[]> upperOrder;
	private boolean m_BackOff;
	protected SmoothingMethod method = SmoothingMethod.HGS;
	private static final int N_INSTANCES_OPTIM_SMOOTHING = 5000;
	final double[] mValues = {0,0.05,0.2,1,5,20};
	private boolean optimizeM = false;

	/**
	 * Build Classifier: Reads the source arff file sequentially and build a
	 * classifier. This incorporated learning the Bayesian network structure and
	 * initializing of the Bayes Tree structure to store the count,
	 * probabilities, gradients and parameters.
	 * 
	 * Once BayesTree structure is initialized, it is populated with the counts
	 * of the data.
	 * 
	 * This is followed by discriminative training using SGD.
	 * 
	 * @param sourceFile
	 * @throws Exception 
	 */
	public void buildClassifier(File sourceFile) throws Exception {

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		m_Instances = structure;
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		paramsPerAtt = new int[nAttributes + 1];// including nc
		for (int u = 0; u < paramsPerAtt.length; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}
		
		bn = new BNStructure(m_Instances, m_S, m_KDB, paramsPerAtt);
		bn.setEnsembleSize(ensembleSize);
		bn.learnStructure(structure, sourceFile, rg);
		
		parentOrderforEachAtt = bn.getParentOrderforEachAtt();
		parentOrder = bn.getLowerOrder();
		upperOrder = bn.getUpperOrder();
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs(); // M-estimation for p(y)
		
//		System.out.println("display all the BNCs");
//		for (int i = 0; i < parentOrder.size(); i++) {
//			System.out.println("upperorder "+i+": "+Arrays.toString(upperOrder.get(i)));
//			for (int j = 0; j < parentOrder.get(i).length; j++) {
//				System.out.print(upperOrder.get(i)[j]+"\t");
//				for (int z = 0; z < parentOrder.get(i)[j].length; z++) {
//					System.out.print(Arrays.toString(parentOrder.get(i)[j][z]) + "\t");
//				}
//				System.out.println();
//			}
//			System.out.println();
//		}
//		
//		for (int i = 0; i < parentOrderforEachAtt.size(); i++) {
//			System.out.println("parent for attribute "+i);
//			for (int j = 0; j < parentOrderforEachAtt.get(i).size(); j++) {
//					System.out.print(Arrays.toString(parentOrderforEachAtt.get(i).get(j).toArray()) + "\t");
//
//			}
//			System.out.println();
//		}
		
//		System.out.println("build tree maps");
		// build trees for each attribute 
		map = new ArrayList<HashMap<ArrayList<Integer>, ProbabilityTree>>();
		for (int u = 0; u < this.nAttributes; u++) {
			map.add(createTreeMap(u));
		}
		
//		for (int i = 0; i < map.size(); i++) {
//			System.out.println("tree for attribute " + i);
//			Set<ArrayList<Integer>> set = map.get(i).keySet();
//			HashMap<ArrayList<Integer>, ProbabilityTree> mapForU = map.get(i);
//
//			for(ProbabilityTree tree:mapForU.values()){
//				System.out.println(tree.getTreeCount());
//			}
////			System.out.println(set.size());
////
////			for (int j = 0; j < set.size(); j++) {
////				System.out.print(Arrays.toString(set.toArray()) + "\t");
////			}
//			 System.out.println();
//		}

//		 dParameters_ = new wdBayesParametersTreePYP_Penny(m_Instances,
//		 paramsPerAtt, parentOrderforEachAtt);

		// System.out.println("\n******************update trees*************");
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			for (int u = 0; u < this.nAttributes; u++) {
				for (ProbabilityTree tempTree : map.get(u).values()) {
					
					int[] parents = tempTree.getParentOrder();
					int nParents = parents.length;
					int[] datapoint = new int[nParents + 1];
					datapoint[0] = (int) row.value(u);
					for (int p = 0; p < nParents; p++) {
						datapoint[1 + p] = (int) row.value(parents[p]);
					}
					tempTree.addObservation(datapoint);
				}
			}
			nInstances++;
		}
		
		switch (method) {
		case M_estimation:
//			System.out.println("M_estimation"); 
			if(optimizeM){
				Instances sampleSmoothing;
	    		if (nInstances / 10 > N_INSTANCES_OPTIM_SMOOTHING) {
	    		    sampleSmoothing = sampleData(sourceFile, N_INSTANCES_OPTIM_SMOOTHING);
	    		} else {
	    		    sampleSmoothing = sampleData(sourceFile, nInstances / 10);
	    		}
	    		
	    		
				// unupdate all the trees
	    		for(Instance ins: sampleSmoothing){
	    			for (int u = 0; u < this.nAttributes; u++) {
	    				for (ProbabilityTree tempTree : map.get(u).values()) {
	    					int[] parentOrder = tempTree.getParentOrder();
	    					
	    					int[] datapoint = new int[parentOrder.length];
	    					for(int i = 0; i < datapoint.length; i++){
	    						datapoint[i] = (int) ins.value(parentOrder[i]);
	    					}
	    					tempTree.unUpdate(ins,u,datapoint);
	    				}
	    			}
	    		}
	    		
	    		double bestM = 1.0;
	    		double bestRMSE = Double.MAX_VALUE;
	    		
	    		for(double m:mValues){
	    			
	    			for (int u = 0; u < this.nAttributes; u++) {
	    				for (ProbabilityTree tempTree : map.get(u).values()) {
	    					SUtils.m_MEsti = m;
	    					tempTree.convertCountToProbs();
	    				}
	    			}
	    			
	    			int nDataPoints = 0;
	    			double rmse = 0;
	    			for (Instance instance : sampleSmoothing) {
	    				int classValue = (int) instance.classValue();
	    				double[] probs = distributionForInstance(instance);
	    				boolean someNaN = false;
	    				
	    				for (int c = 0; !someNaN && c < probs.length; c++) {
	    					if(Double.isNaN(probs[c])){
	    						someNaN = true;
	    					}
	    				}
						
	    				if(!someNaN){
	    					for (int i = 0; i < probs.length; i++) {
	    						if (i == classValue) {
	    							rmse += (1.0 - probs[i]) * (1.0 - probs[i]);
	    						} else {
	    							rmse += (0.0 - probs[i]) * (0.0 - probs[i]);
	    						}
	    					}
	    					nDataPoints++;
	    				}
	    			}
	    			rmse = Math.sqrt(rmse / nDataPoints);
	    			if (rmse < bestRMSE) {
	    				bestRMSE = rmse;
	    				bestM = m;
	    			}
	    			System.out.println("m param="+SUtils.m_MEsti+" RMSE="+rmse+" (best="+bestRMSE+")");
	    		}
	    		SUtils.m_MEsti = bestM;
//	    		System.out.println("Selecting smoothing param="+SUtils.m_MEsti);
	    		
	    		// put back into model
	    		for (Instance instance : sampleSmoothing) {
	    			for (int u = 0; u < this.nAttributes; u++) {
	    				for (ProbabilityTree tempTree : map.get(u).values()) {
	    					int[] parentOrder = tempTree.getParentOrder();
	    					
	    					int[] datapoint = new int[parentOrder.length];
	    					for(int i = 0; i < datapoint.length; i++){
	    						datapoint[i] = (int) instance.value(parentOrder[i]);
	    					}
	    					tempTree.Update(instance,u,datapoint);
	    					tempTree.convertCountToProbs();
	    				}
	    			}
	    		}
			}else{
				for (int u = 0; u < this.nAttributes; u++) {
					for (ProbabilityTree tempTree : map.get(u).values()) {	
						tempTree.convertCountToProbs();
					}
				}
			}

			break;
		case HDP:
//			System.out.println("HDP");
			LogStirlingGenerator lgcache = new LogStirlingGenerator(m_Discount, nInstances);
			for (int u = 0; u < this.nAttributes; u++) {
//				System.out.println("u == " + u);
				for (ProbabilityTree tempTree : map.get(u).values()) {
					// tempTree.setGibbsIteration(m_IterGibbs);
					tempTree.setLogStirlingCache(lgcache);
//					tempTree.setHDP(true);
					tempTree.smooth();
				}
			}
			break;
		case LOOCV:
//			System.out.println("LOOCV");
			for (int u = 0; u < this.nAttributes; u++) {
//				System.out.println("u=="+u);
				for (ProbabilityTree tempTree : map.get(u).values()) {
					tempTree.LOOCV();
//					System.out.println(tempTree.printAlphas());
				}
			}
			
			break;
		default:
			break;
		}
	}
	
	public double[] distributionForInstance(Instance instance) {
		double[] probY = new double[nc];
		for (int c = 0; c < nc; c++) {
			probY[c] = xxyDist_.xyDist_.pp(c);// P(y)
		}
		for (int c = 0; c < nc; c++) {
			for (int u = 0; u < nAttributes; u++) {
				int targetNodeValue = (int) instance.value(this.upperOrder.get(0)[u]);
				int[] parents = this.parentOrder.get(0)[u][0];
				ArrayList<Integer> temp = new ArrayList<Integer>();
				for (int k = 0; k < parents.length; k++) {
					temp.add(parents[k]);
				}
				int[] datapoint = new int[parents.length];
				datapoint[0] = c;
				for (int p = 1; p < parents.length; p++) {
					datapoint[p] = (int) instance.value(parents[p]);
				}
				
				ProbabilityTree tree = map.get(u).get(temp);
				double a = tree.query(datapoint)[targetNodeValue];
				
				probY[c] +=Math.log(a);
			}
		}
		SUtils.normalizeInLogDomain(probY);
		SUtils.exp(probY);
		return probY;
	}
	
//	public double[] distributionForInstance(Instance instance) {
//
//		this.nc=instance.numClasses();
//		double[] probY = new double[nc];
//		for (int c = 0; c < nc; c++) {
//			probY[c] = xxyDist_.xyDist_.pp(c);// P(y)
//		}
//		
////		System.out.println(Arrays.toString(probY));
//		ArrayList<Integer> temp;
//		int targetNodeValue;
//		int[] datapoint;
//		int[] parent;
//		double[] res = new double[nc];
//		
//		for (int i = 0; i < upperOrder.size(); i++) {
//			int[] order = upperOrder.get(i);
//			double[] resForOneUpper = Arrays.copyOf(probY, probY.length);
//			for (int c = 0; c < nc; c++) {
//				for (int u = 0; u < order.length; u++) {
//					
//					double tempResForOneParent = 0;
//					targetNodeValue = (int) instance.value(order[u]);
//					int totalTree = 0;
//					// averaged over multiple parent orders
//					for (int j = 0; j < parentOrder.get(i)[u].length; j++) {
////						
//						parent = parentOrder.get(i)[u][j];
//						temp = new ArrayList<Integer>();
//						for (int k = 0; k < parent.length; k++) {
//							temp.add(parent[k]);
//						}
//						
//						datapoint = new int[parent.length];
//						datapoint[0] = c;
//						for (int p = 1; p < parent.length; p++) {
//							datapoint[p] = (int) instance.value(parent[p]);
//						}
//						ProbabilityTree tree = map.get(order[u]).get(temp);
//						double a = tree.query(datapoint)[targetNodeValue];
//						tempResForOneParent += a;
//					}
//					tempResForOneParent /= parentOrder.get(i)[u].length;
////					System.out.println("tempResForOneParent "+tempResForOneParent);
//					double b = FastMath.log(tempResForOneParent);
////					System.out.println("b == "+b);
//					resForOneUpper[c] += b;
////					System.out.println("res == "+resForOneUpper[c]);
//				}
////				System.out.println("res == "+resForOneUpper[c]);
//			}
//			
//			SUtils.exp(resForOneUpper);
//			SUtils.normalize(resForOneUpper);
////			System.out.println(i+":"+Arrays.toString(resForOneUpper));
//			for (int c = 0; c < nc; c++) {
//				res[c] += resForOneUpper[c];
//			}
//		}
//		
////		System.out.println(Arrays.toString(res));
//		for (int c = 0; c < nc; c++) {
//			res[c] /= upperOrder.size();
//		}
////		System.out.println("all: "+Arrays.toString(res));
////		SUtils.normalize(res);
//		
//		return res;
//	}

	public HashMap<ArrayList<Integer>, ProbabilityTree> createTreeMap(int u) {
		ProbabilityTree tree;
		HashMap<ArrayList<Integer>, ProbabilityTree> mapU = new HashMap<ArrayList<Integer>, ProbabilityTree>();
//		int count = 0;
//		HashMap<ArrayList<Integer>,Integer> allParentforU = parentForEachAtt.get(u);
//		for(HashMap.Entry m : allParentforU.entrySet()){
//			ArrayList<Integer> parentsU = (ArrayList<Integer>) m.getKey();
//			int value = (int) m.getValue();
//			int nParents = parentsU.size();
//			int[] arityConditioningVariables = new int[nParents];
//			for (int p = 0; p < nParents; p++) {
//				arityConditioningVariables[p] = paramsPerAtt[parentsU.get(p)];
//			}
//			tree = new ProbabilityTree(paramsPerAtt[u], arityConditioningVariables, m_IterGibbs, m_Tying);
//			tree.setParentOrder(parentsU);
//			tree.setTreeCount(value);
//			mapU.put(parentsU, tree);
//			count++;
//		}
		
		//		System.out.println(this.parentOrderforEachAtt.get(u).size());
		
		ArrayList<ArrayList<Integer>> allParentforU = parentOrderforEachAtt.get(u);
		for (int i = 0; i < allParentforU.size(); i++) {
			ArrayList<Integer> parentsU = allParentforU.get(i);

			int nParents = parentsU.size();
			if (!mapU.containsKey(parentsU)) {
				// arityConditioningVariables is values for every parent
				// attribute
				int[] arityConditioningVariables = new int[nParents];
				for (int p = 0; p < nParents; p++) {
					arityConditioningVariables[p] = paramsPerAtt[parentsU.get(p)];
				}
				tree = new ProbabilityTree(paramsPerAtt[u], arityConditioningVariables, m_IterGibbs, m_Tying);
				tree.setParentOrder(parentsU);
				mapU.put(parentsU, tree);
			}
		}
		return mapU;
	}

	public String getMS() {
		return m_S;
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

	public Instances getM_Instances() {
		return m_Instances;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public void setK(int m_K) {
		m_KDB = m_K;
	}

	public void set_m_S(String string) {
		m_S = string;
	}

	public void setRandomGenerator(RandomGenerator rg) {
		this.rg = rg;
	}

	public void setM_Iterations(int m) {
		m_IterGibbs = m;
	}

//	public void setMEstimation(boolean m) {
//		M_estimation = m;
//	}

	public void setDiscount(double d) {
		m_Discount = d;
	}

	public void setEnsembleSize(int e) {
		ensembleSize = e;
	}

	public void printAllTrees() {
		for (int i = 0; i < map.size(); i++) {
			for (ProbabilityTree tree : map.get(i).values()) {
				System.out.println(tree.printFinalPks());
			}
		}
	}
	
	public void setM_Tying(int t) {
		m_Tying = t;
	}

	public void setGibbsIteration(int iter) {
		m_IterGibbs = iter;
	}

	public void setBackoff(boolean back) {
		m_BackOff = back;
	}

	public void setMethod(SmoothingMethod method2) {
		// TODO Auto-generated method stub
		this.method = method2;
	}
	
	private Instances sampleData(File sourceFile, int nSamples) throws FileNotFoundException, IOException {
		int nDataPoints = this.nInstances;
		Instances res = new Instances(structure);
		RandomDataGenerator dg = new RandomDataGenerator(rg);
		int[] indexesOfSamples = dg.nextPermutation(nDataPoints, nSamples);
		Arrays.sort(indexesOfSamples);

		Instance row;
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int currentIndex = 0;
		for (int i = 0; i < indexesOfSamples.length; i++) {
			int sampleIndex = indexesOfSamples[i];
			do {
				row = reader.readInstance(structure);
				currentIndex++;
			} while (currentIndex < sampleIndex);
			res.add(row);
		}
		return res;
	}
}
