/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
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
 * Code written by: Nayyar Zaidi, Francois Petitjean
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
import java.util.Iterator;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.gui.SysErrLog;

public class wdBayesOnlineMAPNormalBackoff {

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	
	private Instances structure;

	private xxyDist xxyDist_;
	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private String m_S = "NB"; 											// -S (1: NB, 2:TAN, 3:KDB, 4:BN, 5:Chordalysis, 6:Saturated)
	private String m_P = "MAP"; 											// -P (1: MAP, 2:dCCBN, 3:wCCBN, 4:eCCBN)
	private int m_KDB = 1; 													// -K
	private long m_Chordalysis_Mem = Long.MAX_VALUE; 	// -F (in thousands of free parameters)
	private boolean m_MVerb = false; 									// -V
	private RandomGenerator rg = null;
	
	private BNStructure bn = null;

	private LearningListener learningListener = null;

	private boolean optimizeM = false;

	private static final int N_INSTANCES_OPTIM_SMOOTHING = 5000;
	private static final int BUFFER_SIZE = 100000;

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

		bn = new BNStructure(m_Instances, m_S, m_KDB,paramsPerAtt);
		bn.setMaxNFreeParams(m_Chordalysis_Mem);
		
		bn.learnStructure(structure, sourceFile, rg);

		m_Order = bn.get_Order();
		m_Parents = bn.get_Parents();
		xxyDist_ = bn.get_XXYDist();
		xxyDist_.countsToProbs();

		dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order, m_Parents,0);

		// ------------------------------------------------------------------------
		Instance row;
		this.nInstances = 0;
		while ((row = reader.readInstance(structure)) != null) {
			dParameters_.update(row);
			this.nInstances++;
		}

		dParameters_.countsToProbability();
		
		// Compute initial cost, error and rmse based on the current counts
		if(optimizeM){
        		Instances sampleSmoothing;
        		if (nInstances / 10 > N_INSTANCES_OPTIM_SMOOTHING) {
        		    sampleSmoothing = sampleData(sourceFile, N_INSTANCES_OPTIM_SMOOTHING);
        		} else {
        		    sampleSmoothing = sampleData(sourceFile, nInstances / 10);
        		}
        		dParameters_.optimizeSmoothingParameter(sampleSmoothing, this);
		}
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
	
	public double[] distributionForInstance(Instance instance) {
		double[] probs = logDistributionForInstance_MAP(instance);
		SUtils.exp(probs);
		return probs;
	}

	public double[] logDistributionForInstance_MAP(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = dParameters_.getClassProbabilities()[c]; //xxyDist_.xyDist_.pp(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYProbability((int) instance.value(m_Order[u]),	c);
			}
		}
		
		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	/* 
	 * Miscellaneous functions starting here.
	 */

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

	public wdBayesParametersTree getdParameters_() {
		return dParameters_;
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

	public void setMaxNParameters(long m_Chordalysis_Mem2) {
		this.m_Chordalysis_Mem = m_Chordalysis_Mem2;

	}

	public void setRandomGenerator(RandomGenerator rg) {
		this.rg = rg;
	}


	public void setLearningListener(LearningListener learningListener) {
		this.learningListener = learningListener;
	}
	
	public void setOptimizeM(boolean o){
		this.optimizeM  = o;
	}
}