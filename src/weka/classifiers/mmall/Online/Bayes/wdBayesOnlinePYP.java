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

import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import hdp.ProbabilityTree;
import weka.classifiers.mmall.Bayes.wdBayesParametersTreePYP;
//import weka.classifiers.mmall.Bayes.wdBayesParametersTreePYP;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class wdBayesOnlinePYP {

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;

	private Instances structure;

	private xxyDist xxyDist_;
	public wdBayesParametersTreePYP dParameters_;
	private String m_S = "NB"; 

	private int[][] m_Parents;
	private int[] m_Order;

	private int m_KDB = 1; 													// -K
	private boolean m_MVerb = false; 									// -V
	private RandomGenerator rg = null;

	private BNStructure bn = null;

	private static final int BUFFER_SIZE = 100000;

	int m_BestK_ = 0; 
	int m_BestattIt = 0;
	
	int m_Iterations = 5000;
	int m_Tying = 1;

	/**
	 * Build Classifier: Reads the source arff file sequentially and build a classifier.
	 * This incorporated learning the Bayesian network structure and initializing of the 
	 * Bayes Tree structure to store the count, probabilities, gradients and parameters.
	 * 
	 * Once BayesTree structure is initialized, it is populated with the counts of the data.
	 * 
	 * This is followed by discriminative training using SGD.
	 * 
	 * @param sourceFile
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void buildClassifier(File sourceFile) throws FileNotFoundException, IOException {

//		System.out.println("**** In wdBayesOnlinePYP ****");
//		System.out.println("No. of Iterations = " + m_Iterations);
//		System.out.println("Tying strategy (0 = None. 1 = SAME_PARENT. 2 = LEVEL. 3 = SINGLE) = " + m_Tying);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 10000);
		this.structure = reader.getStructure();
		structure.setClassIndex(structure.numAttributes() - 1);

		m_Instances = structure;
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		bn = new BNStructure(m_Instances, m_S, m_KDB, paramsPerAtt);
		bn.learnStructure(structure, sourceFile, rg);

		m_Order = bn.get_Order();
		m_Parents = bn.get_Parents();
		m_BestattIt = bn.get_BestattIt();
		
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs();

		dParameters_ = new wdBayesParametersTreePYP(m_Instances, paramsPerAtt, m_Order, m_Parents, m_Iterations, m_Tying);

		// ------------------------------------------------------------------------
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
			this.nInstances++;
		}
		
//		dParameters_.prepareForQuerying();
	}

	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			//probs[c] = FastMath.log(xxyDist_.xyDist_.pp(c));// P(y)
			probs[c] = xxyDist_.xyDist_.pp(c);// P(y)
		}
		//		System.out.println("prior"+Arrays.toString(probs));

		for (int u = 0; u < m_BestattIt; u++) {
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

	public int[] getM_Order() {
		return m_Order;
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
	
	public int getM_Iterations() {
		return m_Iterations;
	}

	public void setM_Iterations(int m_Iterations) {
		this.m_Iterations = m_Iterations;
	}

	public void setM_Tying(int m_Tying) {
		this.m_Tying = m_Tying;
	}

}
