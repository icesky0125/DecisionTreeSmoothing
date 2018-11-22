/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ClassifierTree.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.j48;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Vector;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import hdp.logStirling.LogStirlingFactory;
import hdp.logStirling.LogStirlingGenerator;
import hdp.logStirling.LogStirlingGenerator.CacheExtensionException;
import tools.MathUtils;
import tree.ConcentrationC45;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Class for handling a tree structure used for classification.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 11269 $
 */
public class ClassifierTree implements Drawable, Serializable, CapabilitiesHandler, RevisionHandler {

	/** for serialization */
	static final long serialVersionUID = -8722249377542734193L;

	/** The model selection method. */
	protected ModelSelection m_toSelectModel;

	/** Local model at node. */
	protected ClassifierSplitModel m_localModel;

	/** References to sons. */
	public ClassifierTree[] m_sons;

	/** True if node is leaf. */
	public boolean m_isLeaf;

	/** True if node is empty. */
	protected boolean m_isEmpty;

	/** The training instances. */
	protected Instances m_train;

	/** The pruning instances. */
	protected Distribution m_test;

	/** The id for the node. */
	protected int m_id;

	/**
	 * For getting a unique ID when outputting the tree (hashcode isn't guaranteed
	 * unique)
	 */
	private static long PRINTED_NODES = 0;

	/** Alpha-value (for pruning) at the node. */
	public double m_Alpha;

	/**
	 * Number of training examples misclassified by the model (subtree rooted).
	 */
	protected double m_numIncorrectModel;

	/**
	 * Number of training examples misclassified by the model (subtree not rooted).
	 */
	protected double m_numIncorrectTree;

	/** Total number of instances used to build the classifier. */
	protected int m_totalTrainInstances;

	private double m_NotPruning = 0;

	private boolean backoff = true;

	// *************************** For LOOCV
	public double alpha;
	public double[] alc;
	private int nc;
	public double partialDerivative;
	public ArrayList<ClassifierTree> leavesUnderThisNode;
	public double denominator = 0;
	public double[] looDiff;

	// *************************** For HDP
	/**
	 * Max value for sampling TK
	 */
	public static final int MAX_TK = 10000;

	/**
	 * True count
	 */
	public int[] nk;
	/**
	 * sum of nk
	 */
	public int marginal_nk;

	/**
	 * Simulated count
	 */
	public int[] tk;
	/**
	 * sum of tk
	 */
	public int marginal_tk;

	/**
	 * contains the parameters calculated as a function of (c,d,nk,tk)
	 */
	public double[] pk;

	/**
	 * contains the accumulated pk for several runs of Gibbs sampling
	 */
	public double[] pkAveraged;

	/**
	 * contains the number of pks that have been accumulated in the pkSum
	 */
	public int nPkAccumulated;

	public ConcentrationC45 c;

	int varNumberForBanchingChildren;

	public static int windowForSamplingTk = 10;
	double[] probabilityForWindowTk = new double[2 * windowForSamplingTk + 1];
	public ClassifierTree m_parent;
	protected RandomGenerator rng = new MersenneTwister(3071980);

	private int nDatapoints;
	
	double d = 0.0;
	LogStirlingGenerator lgCache;
//	LogStirlingCache lgCache;

	/**
	 * Method for building a classifier tree.
	 * 
	 * @param data the data to build the tree from
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data) throws Exception {

		// can classifier tree handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		nDatapoints = data.numInstances();
		m_totalTrainInstances = data.numInstances();
		buildTree(data, false);
	}

	/**
	 * Builds the tree structure.
	 * 
	 * @param data     the data for which the tree structure is to be generated.
	 * @param keepData is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances data, boolean keepData) throws Exception {
		
		Instances[] localInstances;

		if (keepData) {
			m_train = data;
		}
		
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		
		nc = data.numClasses();
		nk = new int[nc];
		pk = new double[nc];
		this.marginal_nk = 0;
		this.marginal_tk = 0;
		
		this.m_parent = null;
		this.alpha = 1;
		
		for (Instance ins : data) {
			nk[(int) ins.classValue()]++;
		}
		this.marginal_nk = data.numInstances();

		m_localModel = m_toSelectModel.selectModel(data);

		if (m_localModel.numSubsets() > 1) {
			// alpha = 1;
			localInstances = m_localModel.split(data);
			data = null;
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			for (int i = 0; i < m_sons.length; i++) {
				m_sons[i] = getNewTree(localInstances[i]);
				m_sons[i].m_parent = this;
				m_sons[i].m_totalTrainInstances = this.m_totalTrainInstances;
				localInstances[i] = null;
			}
		} else {
			m_isLeaf = true;
			if (Utils.eq(data.sumOfWeights(), 0)) {
				m_isEmpty = true;
			}

			data = null;
		}
	}

	/**
	 * Builds the tree structure with hold out set
	 * 
	 * @param train    the data for which the tree structure is to be generated.
	 * @param test     the test data for potential pruning
	 * @param keepData is training Data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances train, Instances test, boolean keepData) throws Exception {

		Instances[] localTrain, localTest;
		int i;

		if (keepData) {
			m_train = train;
		}
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_localModel = m_toSelectModel.selectModel(train, test);
		m_test = new Distribution(test, m_localModel);
		
		nc = train.numClasses();
		nk = new int[nc];
		pk = new double[nc];
		this.marginal_nk = 0;
		this.marginal_tk = 0;
		
		this.m_parent = null;
		this.alpha = 1;
		
		for (Instance ins : train) {
			nk[(int) ins.classValue()]++;
		}
		this.marginal_nk = train.numInstances();
		
		if (m_localModel.numSubsets() > 1) {
			localTrain = m_localModel.split(train);
			localTest = m_localModel.split(test);
			train = null;
			test = null;
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			for (i = 0; i < m_sons.length; i++) {
				m_sons[i] = getNewTree(localTrain[i], localTest[i]);
				m_sons[i].m_parent = this;
				localTrain[i] = null;
				localTest[i] = null;
			}
		} else {
			m_isLeaf = true;
			if (Utils.eq(train.sumOfWeights(), 0)) {
				m_isEmpty = true;
			}
			train = null;
			test = null;
		}
	}

	/**
	 * Classifies an instance.
	 * 
	 * @param instance the instance to classify
	 * @return the classification
	 * @throws Exception if something goes wrong
	 */
	public double classifyInstance(Instance instance) throws Exception {

		double maxProb = -1;
		double currentProb;
		int maxIndex = 0;
		int j;

		for (j = 0; j < instance.numClasses(); j++) {
			currentProb = getProbs(j, instance, 1);
			if (Utils.gr(currentProb, maxProb)) {
				maxIndex = j;
				maxProb = currentProb;
			}
		}

		return maxIndex;
	}

	/**
	 * Cleanup in order to save memory.
	 * 
	 * @param justHeaderInfo
	 */
	public final void cleanup(Instances justHeaderInfo) {

		m_train = justHeaderInfo;
		m_test = null;
		if (!m_isLeaf) {
			for (ClassifierTree m_son : m_sons) {
				m_son.cleanup(justHeaderInfo);
			}
		}
	}

	/**
	 * Returns class probabilities for a weighted instance.
	 * 
	 * @param instance   the instance to get the distribution for
	 * @param useLaplace whether to use laplace or not
	 * @return the distribution
	 * @throws Exception if something goes wrong
	 */
	public final double[] distributionForInstance(Instance instance, boolean useLaplace) throws Exception {

		double[] doubles = new double[instance.numClasses()];

		for (int i = 0; i < doubles.length; i++) {
			if (!useLaplace) {
				doubles[i] = getProbs(i, instance, 1);
			} else {
				doubles[i] = getProbsLaplace(i, instance, 1);
			}
		}

		return doubles;
	}

	public final double[] distributionForInstance(Instance instance) throws Exception {

		double[] doubles = new double[instance.numClasses()];
		
		

		for (int i = 0; i < doubles.length; i++) {
			doubles[i] = getProbs(i, instance, 1);
		}
//		System.out.println(Utils.sum(doubles));

		return doubles;
	}

	public void setLogStirlingCache(LogStirlingGenerator cache) {
		this.lgCache = cache;

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].setLogStirlingCache(cache);
				}
			}
		}
	}

	public void prepareForSamplingTk() {
		if (tk == null) {
			tk = new int[nk.length];
		}

		if (m_sons != null) {
			// first we launch the recursive call to make the nk and
			// tk correct for the children
			for (int c = 0; m_sons != null && c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].prepareForSamplingTk();
				}
			}

			/*
			 * Now the tks (and nks) from the children are correctly set. If a leaf, nk is
			 * already set, so we only have to do it if not a leaf (by summing the tks from
			 * the children).
			 */
			for (int k = 0; k < nk.length; k++) {
				nk[k] = 0;
			}
			marginal_nk = 0;

			for (int c = 0; m_sons != null && c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					for (int k = 0; k < nk.length; k++) {
						int tkChild = m_sons[c].tk[k];
						nk[k] += tkChild;
						marginal_nk += tkChild;
					}
				}
			}
		}

		// Now nks are set for current node; let's initialize the tks
		if (this.m_parent == null) {
			for (int k = 0; k < nk.length; k++) {
				tk[k] = (nk[k] == 0) ? 0 : 1;
				marginal_tk += tk[k];
			}
			// System.out.println(Arrays.toString(nk) + "\t" +
			// Arrays.toString(tk));
		} else {
			double concentration = getConcentration();
			for (int k = 0; k < nk.length; k++) {
				if (nk[k] <= 1) {
					tk[k] = nk[k];
				} else if (d > 0) {// PYP case
					tk[k] = (int) Math.max(1, Math.floor(concentration / d
							* (getRisingFact(concentration + d, nk[k]) / getRisingFact(concentration, nk[k]) - 1)));
				} else {// DP case
					tk[k] = (int) Math.max(1,
							Math.floor(concentration * (digamma(concentration + nk[k]) - digamma(concentration))));
				}
				marginal_tk += tk[k];
			}
			// System.out.println(Arrays.toString(nk) + "\t" +
			// Arrays.toString(tk));
		}
	}

	public double getConcentration() {
		if (c == null) {
			return 2.0;
		} else {
			return c.c;
		}
	}

	public void setConcentration(double alpha) {
		this.c.c = alpha;
	}

	public double getRisingFact(double x, int n) {
		return FastMath.exp(MathUtils.logGammaRatio(x, n));
	}

	public double digamma(double d) {
		return Gamma.digamma(d);
	}

	public void setDiscountRecursively(double d) {
		this.d = d;

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].setDiscountRecursively(d);
				}
			}
		}
	}

	public void sampleTks() {
		// System.out.println("tk="+Arrays.toString(tk));
		if (m_parent == null) {
			/*
			 * case for root: no sampling, t is either 0 or 1
			 */
			// System.out.println("sampling root");
			for (int k = 0; k < tk.length; k++) {
				// Wray says this is GEM
				int t = (nk[k] == 0) ? 0 : 1;
				setTk(k, t);
			}
		} else {
			for (int k = 0; k < tk.length; k++) {

				// String treeBefore = tree.printTksAndNks();
				// System.out.println("starting score = "+tree.logScoreTree()+"
				// with tk="+Arrays.toString(tk));
				// System.out.println("previous tk["+k+"]="+oldTk);
				if (nk[k] <= 1) {
					/*
					 * can't sample anything, constraints say that tk[k] must be nk[k] just have to
					 * check that tk[k] is different or not to the previous time (in case nk[k] has
					 * just changed)
					 */
					setTk(k, nk[k]);
					// System.out.println("can't sample; should be set to " +
					// nk[k] + " tk=" + tk[k]);
				} else {
					// sample case
					// starting point
					int oldTk = tk[k];
					int valTk = tk[k] - windowForSamplingTk;
					// maxTk can't be larger than nk[k]
					int maxTk = Math.min(tk[k] + windowForSamplingTk, nk[k]);

					// Limit maxTk for big dataset
					if (maxTk > MAX_TK) {
						maxTk = MAX_TK;
					}

					int index = 0;
					while (valTk < 1) {// move to first allowed position
						probabilityForWindowTk[index] = Double.NEGATIVE_INFINITY;
						valTk++;
						index++;
					}
					boolean hasOneValue = false;
					while (valTk <= maxTk) {// now fill posterior
						double logProbDifference = setTk(k, valTk);
						probabilityForWindowTk[index] = logProbDifference;
						hasOneValue = (hasOneValue || probabilityForWindowTk[index] != Double.NEGATIVE_INFINITY);
						index++;
						valTk++;
					}
					if (!hasOneValue) {
						setTk(k, oldTk);
						continue;
					}
					for (; index < probabilityForWindowTk.length; index++) {
						// finish filling with neg infty
						probabilityForWindowTk[index] = Double.NEGATIVE_INFINITY;
					}

					// now lognormalize probabilityForWindowTk and exponentiate
					MathUtils.normalizeInLogDomain(probabilityForWindowTk);
					MathUtils.exp(probabilityForWindowTk);

					for (int j = 0; j < probabilityForWindowTk.length; j++) {
						if (Double.isNaN(probabilityForWindowTk[j])) {
							System.err.println("problem " + Arrays.toString(probabilityForWindowTk));
						}
					}
					// System.out.println(Arrays.toString(probabilityForWindowTk));

					// now sampling tk according to probability vector
					int chosenIndex = MathUtils.sampleFromMultinomial(rng, probabilityForWindowTk);

					// assign chosen tk
					int valueTkChosen = oldTk - windowForSamplingTk + chosenIndex;
					setTk(k, valueTkChosen);
				}
			}
		}
	}

	protected double setTk(int k, int val) {
		// how much to increment (or decrement tk by)
		int incVal = val - tk[k];
		if (incVal < 0) {
			// if decrement, then have to check that valid for the
			// parent
			if (m_parent != null && incVal < 0 && (m_parent.nk[k] + incVal) < m_parent.tk[k]) {
				// not valid; skip
				return Double.NEGATIVE_INFINITY;
			}
		}

		tk[k] += incVal;
		marginal_tk += incVal;

		double concentration = getConcentration();
		double res = 0.0;

		// partial score difference for current node
		try {
			res += logStirling(0.0, nk[k], tk[k]);
		} catch (CacheExtensionException e) {
			System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
			e.printStackTrace();
			System.exit(1);
		}

		res += MathUtils.logPochhammerSymbol(concentration, d, marginal_tk);

		// partial score difference for parent
		if (m_parent != null) {
			m_parent.nk[k] += incVal;
			m_parent.marginal_nk += incVal;

			try {
				res += logStirling(0.0, m_parent.nk[k], m_parent.tk[k]);
			} catch (CacheExtensionException e) {
				System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
				e.printStackTrace();
				System.exit(1);
			}

			// res -= MathUtils.logGammaRatio(parent.getConcentration(),
			// parent.marginal_nk);
			res -= m_parent.c.logGammaRatioForConcentration(m_parent.marginal_nk);
		}

		return res;
		// return tree.logScoreTree();
	}

	protected double logStirling(double a, int n, int m) throws CacheExtensionException {
		if (a != lgCache.discountP) {
			try {
				// Do not forget to close to free resources!
				lgCache.close();
			} catch (Exception e) {
				System.err.println("Closing Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}

			try {
				lgCache = LogStirlingFactory.newLogStirlingGenerator(nDatapoints, a);
			} catch (NoSuchFieldException | IllegalAccessException e) {
				System.err.println("Log Stirling Cache Exception " + e.getMessage());
				System.err.println("Throws as RuntimeException");
				throw new RuntimeException(e);
			}
		}

		double res = lgCache.query(n, m);
		return res;
	}

	/**
	 * This function computes the values of the smoothed conditional probabilities
	 * as a function of (nk,tk,c,d) and of the parent probability. <br/>
	 * p_k = ( ( nk - tk*d ) / (N + c) ) ) + ( ( c + T*d ) / (N + c) ) ) *
	 * p^{parent}_k
	 * 
	 * @see <a href=
	 *      "http://topicmodels.org/2014/11/13/training-a-pitman-yor-process-tree-with-observed-data-at-the-leaves-part-2/">
	 *      topicmodels.org</a> (Equation 1)
	 */
	public void computeProbabilities() {
		if (pk == null) {
			pk = new double[nk.length];
		}
		
		double concentration = getConcentration();
		double sum = 0.0;
		for (int k = 0; k < pk.length; k++) {
			double parentProb = (m_parent != null) ? m_parent.pk[k] : 1.0 / pk.length;// uniform
																						// parent
																						// if
																						// root
																						// node
			pk[k] = nk[k] / (marginal_nk + concentration) + concentration * parentProb / (marginal_nk + concentration);
			sum += pk[k];
		}

		// normalize
		for (int k = 0; k < pk.length; k++) {
			pk[k] /= sum;
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].computeProbabilities();
				}
			}
		}
	}

	/**
	 * This method accumulates the pks so that the final result is averaged over
	 * several successive iterations of the Gibbs sampling process in log space to
	 * avoid underflow
	 */
	public void recordAndAverageProbabilities() {
		// in this method, pkAveraged stores the log sum
		if (pkAveraged == null) {
			pkAveraged = new double[nk.length];
			nPkAccumulated = 1;
		}
		
		double sum = 0.0;
		for (int k = 0; k < pkAveraged.length; k++) {
			pkAveraged[k] += (pk[k] - pkAveraged[k]) / nPkAccumulated;
			sum += pkAveraged[k];
		}

		// normalize
		for (int k = 0; k < pk.length; k++) {
			pkAveraged[k] /= sum;
		}
		nPkAccumulated++;

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].recordAndAverageProbabilities();
				}
			}
		}
	}

	public double logScoreSubTree() {
		double res = 0.0;
		double concentration = getConcentration();
		res += MathUtils.logPochhammerSymbol(concentration, d, marginal_tk);
		res -= c.logGammaRatioForConcentration(marginal_nk);

		// Now nks are set for current node; let's initialize the tks
		for (int k = 0; k < nk.length; k++) {

			try {
				res += logStirling(0.0, nk[k], tk[k]);
			} catch (CacheExtensionException e) {
				System.err.println("Cannot extends the cache to querry S(" + nk[k] + ", " + tk[k] + ")");
				e.printStackTrace();
				System.exit(1);
			}
			if (res == Double.NEGATIVE_INFINITY) {
				throw new RuntimeException("log stirling return neg infty");
			}
		}
		// we score all of the children (doesn't matter if done first or after)
		for (int c = 0; m_sons != null && c < m_sons.length; c++) {
			if (m_sons[c] != null) {
				// System.out.println(m_sons[c].m_isEmpty);
				res += m_sons[c].logScoreSubTree();
			}
		}
		return res;
	}

	public ArrayList<ClassifierTree> getAllNodes() {
		ArrayList<ClassifierTree> res = new ArrayList<ClassifierTree>();
		res.add(this);

		if (m_sons != null) {
			for (ClassifierTree node : m_sons) {
				if (node != null) {
					res.addAll(node.getAllNodes());
				}
			}
		}
		return res;
	}

	public ArrayList<ClassifierTree> getAllNodesAtRelativeDepth(int depth) {
		ArrayList<ClassifierTree> res = new ArrayList<>();
		if (depth == 0) {
			res.add(this);
		} else {
			if (m_sons != null) {
				for (int c = 0; c < m_sons.length; c++) {
					if (m_sons[c] != null) {
						res.addAll(m_sons[c].getAllNodesAtRelativeDepth(depth - 1));
					}
				}
			}
		}
		return res;
	}

	public String printPksRecursively(String prefix) {
		String res = "";

		res += prefix + ":pk=" + Arrays.toString(this.pkAveraged) + " c=" + this.c + "\n";
		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printPksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printNks(String prefix) {
		String res = "";
		if (m_isLeaf) {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + this.m_Alpha + "\n";

		} else {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + this.m_Alpha + "\t " + this.m_NotPruning + "\n";
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printNks(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printTksAndNksRecursively(String prefix) {
		String res = "";

		// root node
		res += prefix + " :nk=" + Arrays.toString(nk) + ":tk=" + Arrays.toString(tk) + " :pk="
				+ Arrays.toString(this.pkAveraged) + " :c=" + Utils.doubleToString(this.c.c, 4) + "\n";

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printTksAndNksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public int numInnerNodes() {
		int numNodes = 0;
		if (!this.m_isLeaf) {
			numNodes++;
			for (ClassifierTree m_Successor : m_sons) {
				numNodes += m_Successor.numInnerNodes();
			}
		}

		return numNodes;
	}

	public void modelErrors() {
		m_numIncorrectModel = Utils.sum(this.nk) - nk[Utils.maxIndex(this.nk)];
		// m_numIncorrectModel = SUtils.minAbsValueInAnArray(this.nk);

		if (this.m_sons != null) {
			for (ClassifierTree m_Successor : m_sons) {
				if (m_Successor != null)
					m_Successor.modelErrors();
			}
		}
	}

	public Vector<ClassifierTree> getInnerNodes() {

		Vector<ClassifierTree> nodeList = new Vector<ClassifierTree>();
		if (!m_isLeaf) {
			nodeList.add(this);
			for (ClassifierTree son : this.m_sons) {
				if (son != null) {
					nodeList.addAll(son.getInnerNodes());
				}
			}
		}
		return nodeList;
	}

	public void treeErrors() {

		if (m_isLeaf) {
			m_numIncorrectTree = m_numIncorrectModel;
		} else {
			m_numIncorrectTree = 0;
			for (ClassifierTree m_Successor : this.m_sons) {
				m_Successor.treeErrors();
				m_numIncorrectTree += m_Successor.m_numIncorrectTree;
			}
		}
	}

	public void calculateAlphas() {

		if (!m_isLeaf) {
			double errorDiff = m_numIncorrectModel - m_numIncorrectTree;
			if (errorDiff <= 0) {
				// split increases training error (should not normally happen).
				// prune it instantly.
				makeLeaf();
				m_Alpha = Double.MAX_VALUE;
			} else {
				// compute alpha
				errorDiff /= this.m_totalTrainInstances;
				m_Alpha = errorDiff / (numLeaves() - 1);
				long alphaLong = Math.round(m_Alpha * Math.pow(10, 10));
				m_Alpha = alphaLong / Math.pow(10, 10);
				for (ClassifierTree m_Successor : m_sons) {
					m_Successor.calculateAlphas();
				}
			}
		} else {
			// alpha = infinite for leaves (do not want to prune)
			m_Alpha = Double.MAX_VALUE;
		}
	}

	public void makeLeaf() {
		this.m_isLeaf = true;
	}

	public void unprune() {
		if (this.m_sons != null) {
			m_isLeaf = false;
			for (ClassifierTree m_Successor : m_sons) {
				m_Successor.unprune();
			}
		}
	}

	public void expectedErrorForAllLeaves() {

		int sum = Utils.sum(nk);
		int max = nk[Utils.maxIndex(nk)];
		this.m_Alpha = (double) (sum - max + nc - 1) / (sum + nc);

		if (!m_isLeaf) {

			for (ClassifierTree node : this.m_sons) {
				if (node != null) {
					node.expectedErrorForAllLeaves();
				}
			}
		}
	}

	public void minimumErrorPruning() {

		if (!this.m_isLeaf) {
			int sum = Utils.sum(this.nk);
			this.m_NotPruning = 0;

			for (ClassifierTree son : this.m_sons) {
				if (son != null) {
					son.minimumErrorPruning();
					this.m_NotPruning += (double) Utils.sum(son.nk) / sum * son.m_NotPruning;
				}
			}

			if (this.m_Alpha < this.m_NotPruning) {
				this.m_isLeaf = true;
				this.m_sons = null;

				this.m_NotPruning = this.m_Alpha;
			}

		} else {
			this.m_NotPruning = this.m_Alpha;
		}
	}

	public void convertCountToProbsMestimation() {

		this.pkAveraged = new double[nk.length];
		if (Utils.sum(nk) != 0) {
			for (int i = 0; i < nk.length; i++) {
				pkAveraged[i] = SUtils.MEsti(nk[i], marginal_nk, nk.length);
			}
		} else {
			if (this.backoff) {
				pkAveraged = this.m_parent.pkAveraged;
			} else {
				for (int i = 0; i < nk.length; i++) {
					pkAveraged[i] = (double) 1 / nk.length;
				}
			}
		}

		if (this.m_sons != null) {
			for (int i = 0; i < m_sons.length; i++) {
				if (m_sons[i] != null) {
					m_sons[i].convertCountToProbsMestimation();
				}
			}
		}
	}

	public void setPkAccumulatedToPk() {
		if (pkAveraged == null) {
			pkAveraged = new double[nk.length];
		}

		for (int k = 0; k < pkAveraged.length; k++) {
			pkAveraged[k] = pk[k];
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					m_sons[c].setPkAccumulatedToPk();
				}
			}
		}
	}

	public void calculateLOOestimatesTopDown(double[] parentProbs, double parentAlpha) {

		if (this.pk == null) {
			this.pk = new double[nc];
		}

		for (int c = 0; c < nc; c++) {
			if (nk[c] >= 1) {
				pk[c] = (nk[c] - 1 + parentAlpha * parentProbs[c]) / (marginal_nk - 1 + parentAlpha);
			}
		}

		this.denominator = (double) this.alpha / (this.marginal_nk - 1 + parentAlpha);
		this.looDiff = new double[nc];
		for (int c = 0; c < nc; c++) {
			this.looDiff[c] = this.pk[c] - parentProbs[c];
		}

		if (this.m_sons != null) {
			for (int s = 0; s < m_sons.length; s++) {
				if (m_sons[s] != null) {
					m_sons[s].calculateLOOestimatesTopDown(pk, this.alpha);
				}
			}
		}
	}

	public void calculatePKforLeavesTopDown(double[] parentProbs, double parentAlpha) {

		if (this.pkAveraged == null) {
			this.pkAveraged = new double[nc];
		}

		for (int c = 0; c < nc; c++) {
			pkAveraged[c] = (nk[c] + parentAlpha * parentProbs[c]) / (marginal_nk + parentAlpha);
		}

		Utils.normalize(this.pkAveraged);

		if (this.m_sons != null) {
			for (int s = 0; s < m_sons.length; s++) {
				if (m_sons[s] != null) {
					m_sons[s].calculatePKforLeavesTopDown(pkAveraged, this.alpha);
				}
			}
		}
	}

	public double calculatePartialDerivativeDownUp(int c) {

		if (this.m_sons != null) {
			double a = 0;
			double temp = 0;
			double temp1 = 0;
			double b = 0;
			for (int s = 0; s < this.m_sons.length; s++) {
				ClassifierTree childNode = this.m_sons[s];
				if (childNode != null && childNode.marginal_nk != 0) {
					double leaf = childNode.calculatePartialDerivativeDownUp(c);

					if (childNode.m_isLeaf) {
						childNode.alpha = 1;
					}
					temp = leaf * childNode.denominator * childNode.alpha;
					temp1 = temp * childNode.looDiff[c];

					b += temp;
					a += temp1; // sum of partial deriviative of its children
				}
			}

			this.partialDerivative += a;
			return b;
		} else {
			// leaf node
			double pp = this.nk[c] * (1 - this.pk[c]);
			return pp;
		}
	}

	public void calculatePartialDerivativeDownUp() {

		for (int c = 0; c < nc; c++) {
			this.calculatePartialDerivativeDownUp(c);
		}
	}

	public void convertAlphaToConcentrations() {

		if (this.m_sons != null) {
			for (int i = 0; i < this.m_sons.length; i++) {
				m_sons[i].setConcentration(this.alpha);
				m_sons[i].convertAlphaToConcentrations();
			}
		}

	}

	// ******************************** Methods for Validation smoothing
	// ****************
	public void clearCounts() {
		this.nk = new int[nc];
		this.marginal_nk = 0;
		if (this.m_sons != null) {
			for (int i = 0; i < this.m_sons.length; i++) {
				m_sons[i].clearCounts();
			}
		}
	}

	public void learnParameter(Instances validateSet) throws Exception {
		for (int z = 0; z < validateSet.numInstances(); z++) {
			Instance instance = validateSet.instance(z);
			this.updateTreeWithNewInstance(instance);

		}
	}

	private void updateTreeWithNewInstance(Instance instance) throws Exception {

		int classIndex = (int) instance.classValue();
		this.nk[classIndex]++;
		this.marginal_nk++;

		if (this.m_isLeaf) {
			return;
		} else {
			int treeIndex = localModel().whichSubset(instance);

			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						son(i).updateTreeWithNewInstance(instance);
					}
				}
			} else {
				if (!son(treeIndex).m_isEmpty) {
					son(treeIndex).updateTreeWithNewInstance(instance);
				}
			}
		}

	}

	public void convertCountToProbs(String s) {

		// TODO Auto-generated method stub
		pkAveraged = new double[nk.length];
		if (Utils.sum(nk) != 0) {
			for (int i = 0; i < nk.length; i++) {
				if (s.equals("LAPLACE")) {
					pkAveraged[i] = SUtils.Laplace(nk[i], marginal_nk, nk.length);
				} else if (s.equals("M_estimation")) {
					pkAveraged[i] = SUtils.MEsti(nk[i], marginal_nk, nk.length);
				} else if (s.equals("None")) {
					if (nk[i] == 0 || marginal_nk == 0) {
						pkAveraged[i] = 0;
					} else {
						pkAveraged[i] = (double) nk[i] / marginal_nk;
					}
				}
			}
		} else {
			if (this.backoff) {
				pkAveraged = this.m_parent.pkAveraged;
			} else {
				for (int i = 0; i < nk.length; i++) {
					pkAveraged[i] = (double) 1 / nk.length;
				}
			}
		}

		if (this.m_sons != null) {
			for (int i = 0; i < m_sons.length; i++) {
				if (m_sons[i] != null) {
					m_sons[i].convertCountToProbs(s);
				}
			}
		}
	}
	
	// ******************************** Methods by C45 decision tree
	// ****************
	/**
	 * Gets the next unique node ID.
	 * 
	 * @return the next unique node ID.
	 */
	protected static long nextID() {

		return PRINTED_NODES++;
	}

	/**
	 * Resets the unique node ID counter (e.g. between repeated separate print
	 * types)
	 */
	protected static void resetID() {

		PRINTED_NODES = 0;
	}

	/**
	 * Constructor.
	 */
	public ClassifierTree(ModelSelection toSelectLocModel) {

		m_toSelectModel = toSelectLocModel;
	}

	/**
	 * Returns default capabilities of the classifier tree.
	 * 
	 * @return the capabilities of this classifier tree
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = new Capabilities(this);
		result.enableAll();

		return result;
	}

	/**
	 * Assigns a uniqe id to every node in the tree.
	 * 
	 * @param lastID the last ID that was assign
	 * @return the new current ID
	 */
	public int assignIDs(int lastID) {

		int currLastID = lastID + 1;

		m_id = currLastID;
		if (m_sons != null) {
			for (ClassifierTree m_son : m_sons) {
				currLastID = m_son.assignIDs(currLastID);
			}
		}
		return currLastID;
	}

	/**
	 * Returns the type of graph this classifier represents.
	 * 
	 * @return Drawable.TREE
	 */
	@Override
	public int graphType() {
		return Drawable.TREE;
	}

	/**
	 * Returns graph describing the tree.
	 * 
	 * @throws Exception if something goes wrong
	 * @return the tree as graph
	 */
	@Override
	public String graph() throws Exception {

		StringBuffer text = new StringBuffer();

		assignIDs(-1);
		text.append("digraph J48Tree {\n");
		if (m_isLeaf) {
			text.append("N" + m_id + " [label=\"" + Utils.backQuoteChars(m_localModel.dumpLabel(0, m_train)) + "\" "
					+ "shape=box style=filled ");
			if (m_train != null && m_train.numInstances() > 0) {
				text.append("data =\n" + m_train + "\n");
				text.append(",\n");

			}
			text.append("]\n");
		} else {
			text.append("N" + m_id + " [label=\"" + Utils.backQuoteChars(m_localModel.leftSide(m_train)) + "\" ");
			if (m_train != null && m_train.numInstances() > 0) {
				text.append("data =\n" + m_train + "\n");
				text.append(",\n");
			}
			text.append("]\n");
			graphTree(text);
		}

		return text.toString() + "}\n";
	}

	/**
	 * Returns tree in prefix order.
	 * 
	 * @throws Exception if something goes wrong
	 * @return the prefix order
	 */
	public String prefix() throws Exception {

		StringBuffer text;

		text = new StringBuffer();
		if (m_isLeaf) {
			text.append("[" + m_localModel.dumpLabel(0, m_train) + "]");
		} else {
			prefixTree(text);
		}

		return text.toString();
	}

	/**
	 * Returns source code for the tree as an if-then statement. The class is
	 * assigned to variable "p", and assumes the tested instance is named "i". The
	 * results are returned as two stringbuffers: a section of code for assignment
	 * of the class, and a section of code containing support code (eg: other
	 * support methods).
	 * 
	 * @param className the classname that this static classifier has
	 * @return an array containing two stringbuffers, the first string containing
	 *         assignment code, and the second containing source for support code.
	 * @throws Exception if something goes wrong
	 */
	public StringBuffer[] toSource(String className) throws Exception {

		StringBuffer[] result = new StringBuffer[2];
		if (m_isLeaf) {
			result[0] = new StringBuffer("    p = " + m_localModel.distribution().maxClass(0) + ";\n");
			result[1] = new StringBuffer("");
		} else {
			StringBuffer text = new StringBuffer();
			StringBuffer atEnd = new StringBuffer();

			long printID = ClassifierTree.nextID();

			text.append("  static double N").append(Integer.toHexString(m_localModel.hashCode()) + printID)
					.append("(Object []i) {\n").append("    double p = Double.NaN;\n");

			text.append("    if (").append(m_localModel.sourceExpression(-1, m_train)).append(") {\n");
			text.append("      p = ").append(m_localModel.distribution().maxClass(0)).append(";\n");
			text.append("    } ");
			for (int i = 0; i < m_sons.length; i++) {
				text.append("else if (" + m_localModel.sourceExpression(i, m_train) + ") {\n");
				if (m_sons[i].m_isLeaf) {
					text.append("      p = " + m_localModel.distribution().maxClass(i) + ";\n");
				} else {
					StringBuffer[] sub = m_sons[i].toSource(className);
					text.append(sub[0]);
					atEnd.append(sub[1]);
				}
				text.append("    } ");
				if (i == m_sons.length - 1) {
					text.append('\n');
				}
			}

			text.append("    return p;\n  }\n");

			result[0] = new StringBuffer("    p = " + className + ".N");
			result[0].append(Integer.toHexString(m_localModel.hashCode()) + printID).append("(i);\n");
			result[1] = text.append(atEnd);
		}
		return result;
	}

	/**
	 * Returns number of leaves in tree structure.
	 * 
	 * @return the number of leaves
	 */
	public int numLeaves() {

		int num = 0;
		int i;

		if (m_isLeaf) {
			return 1;
		} else {
			for (i = 0; i < m_sons.length; i++) {
				num = num + m_sons[i].numLeaves();
			}
		}

		return num;
	}

	/**
	 * Returns number of nodes in tree structure.
	 * 
	 * @return the number of nodes
	 */
	public int numNodes() {

		int no = 1;
		int i;

		if (!m_isLeaf) {
			for (i = 0; i < m_sons.length; i++) {
				no = no + m_sons[i].numNodes();
			}
		}

		return no;
	}

	/**
	 * Prints tree structure.
	 * 
	 * @return the tree structure
	 */
	@Override
	public String toString() {

		try {
			StringBuffer text = new StringBuffer();

			if (m_isLeaf) {
				text.append(": ");
				text.append(m_localModel.dumpLabel(0, m_train));
			} else {
				dumpTree(0, text);
			}
			text.append("\n\nNumber of Leaves  : \t" + numLeaves() + "\n");
			text.append("\nSize of the tree : \t" + numNodes() + "\n");

			return text.toString();
		} catch (Exception e) {
			return "Can't print classification tree.";
		}
	}

	/**
	 * Returns a newly created tree.
	 * 
	 * @param data the training data
	 * @return the generated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances data) throws Exception {

		ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
		newTree.buildTree(data, false);

		return newTree;
	}

	/**
	 * Returns a newly created tree.
	 * 
	 * @param train the training data
	 * @param test  the pruning data.
	 * @return the generated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances train, Instances test) throws Exception {

		ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
		newTree.buildTree(train, test, false);

		return newTree;
	}

	/**
	 * Help method for printing tree structure.
	 * 
	 * @param depth the current depth
	 * @param text  for outputting the structure
	 * @throws Exception if something goes wrong
	 */
	private void dumpTree(int depth, StringBuffer text) throws Exception {

		int i, j;

		for (i = 0; i < m_sons.length; i++) {
			text.append("\n");
			;
			for (j = 0; j < depth; j++) {
				text.append("|   ");
			}
			text.append(m_localModel.leftSide(m_train));
			text.append(m_localModel.rightSide(i, m_train));
			if (m_sons[i].m_isLeaf) {
				text.append(": ");
				text.append(m_localModel.dumpLabel(i, m_train));
			} else {
				m_sons[i].dumpTree(depth + 1, text);
			}
		}
	}

	/**
	 * Help method for printing tree structure as a graph.
	 * 
	 * @param text for outputting the tree
	 * @throws Exception if something goes wrong
	 */
	private void graphTree(StringBuffer text) throws Exception {

		for (int i = 0; i < m_sons.length; i++) {
			text.append("N" + m_id + "->" + "N" + m_sons[i].m_id + " [label=\""
					+ Utils.backQuoteChars(m_localModel.rightSide(i, m_train).trim()) + "\"]\n");
			if (m_sons[i].m_isLeaf) {
				text.append("N" + m_sons[i].m_id + " [label=\""
						+ Utils.backQuoteChars(m_localModel.dumpLabel(i, m_train)) + "\" " + "shape=box style=filled ");
				if (m_train != null && m_train.numInstances() > 0) {
					text.append("data =\n" + m_sons[i].m_train + "\n");
					text.append(",\n");
				}
				text.append("]\n");
			} else {
				text.append("N" + m_sons[i].m_id + " [label=\""
						+ Utils.backQuoteChars(m_sons[i].m_localModel.leftSide(m_train)) + "\" ");
				if (m_train != null && m_train.numInstances() > 0) {
					text.append("data =\n" + m_sons[i].m_train + "\n");
					text.append(",\n");
				}
				text.append("]\n");
				m_sons[i].graphTree(text);
			}
		}
	}

	/**
	 * Prints the tree in prefix form
	 * 
	 * @param text the buffer to output the prefix form to
	 * @throws Exception if something goes wrong
	 */
	private void prefixTree(StringBuffer text) throws Exception {

		text.append("[");
		text.append(m_localModel.leftSide(m_train) + ":");
		for (int i = 0; i < m_sons.length; i++) {
			if (i > 0) {
				text.append(",\n");
			}
			text.append(m_localModel.rightSide(i, m_train));
		}
		for (int i = 0; i < m_sons.length; i++) {
			if (m_sons[i].m_isLeaf) {
				text.append("[");
				text.append(m_localModel.dumpLabel(i, m_train));
				text.append("]");
			} else {
				m_sons[i].prefixTree(text);
			}
		}
		text.append("]");
	}

	/**
	 * Help method for computing class probabilities of a given instance.
	 * 
	 * @param classIndex the class index
	 * @param instance   the instance to compute the probabilities for
	 * @param weight     the weight to use
	 * @return the laplace probs
	 * @throws Exception if something goes wrong
	 */
	private double getProbsLaplace(int classIndex, Instance instance, double weight) throws Exception {

		double prob = 0;

		if (m_isLeaf) {
			return weight * localModel().classProbLaplace(classIndex, instance, -1);
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						prob += son(i).getProbsLaplace(classIndex, instance, weights[i] * weight);
					}
				}
				return prob;
			} else {
				if (son(treeIndex).m_isEmpty) {
					return weight * localModel().classProbLaplace(classIndex, instance, treeIndex);
				} else {
					return son(treeIndex).getProbsLaplace(classIndex, instance, weight);
				}
			}
		}
	}

	/**
	 * Help method for computing class probabilities of a given instance.
	 * 
	 * @param classIndex the class index
	 * @param instance   the instance to compute the probabilities for
	 * @param weight     the weight to use
	 * @return the probs
	 * @throws Exception if something goes wrong
	 */
	private double getProbs(int classIndex, Instance instance, double weight) throws Exception {

		double prob = 0;

		if (m_isLeaf) {
			return weight * this.pkAveraged[classIndex];
			// return this.pk[classIndex];
//			 return weight * localModel().classProb(classIndex, instance, -1); // no smoothing
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						prob += son(i).getProbs(classIndex, instance, weights[i] * weight);
					}
				}
				return prob;
//				return this.pkAveraged[classIndex];
			} else {
				if (son(treeIndex).m_isEmpty) {
					return weight * localModel().classProb(classIndex, instance, treeIndex);
				} else {
					return son(treeIndex).getProbs(classIndex, instance, weight);
				}
			}
		}
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	private ClassifierSplitModel localModel() {

		return m_localModel;
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	private ClassifierTree son(int index) {

		return m_sons[index];
	}

	/**
	 * Computes a list that indicates node membership
	 */
	public double[] getMembershipValues(Instance instance) throws Exception {

		// Set up array for membership values
		double[] a = new double[numNodes()];

		// Initialize queues
		Queue<Double> queueOfWeights = new LinkedList<Double>();
		Queue<ClassifierTree> queueOfNodes = new LinkedList<ClassifierTree>();
		queueOfWeights.add(instance.weight());
		queueOfNodes.add(this);
		int index = 0;

		// While the queue is not empty
		while (!queueOfNodes.isEmpty()) {

			a[index++] = queueOfWeights.poll();
			ClassifierTree node = queueOfNodes.poll();

			// Is node a leaf?
			if (node.m_isLeaf) {
				continue;
			}

			// Which subset?
			int treeIndex = node.localModel().whichSubset(instance);

			// Space for weight distribution
			double[] weights = new double[node.m_sons.length];

			// Check for missing value
			if (treeIndex == -1) {
				weights = node.localModel().weights(instance);
			} else {
				weights[treeIndex] = 1.0;
			}
			for (int i = 0; i < node.m_sons.length; i++) {
				queueOfNodes.add(node.son(i));
				queueOfWeights.add(a[index - 1] * weights[i]);
			}
		}
		return a;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 11269 $");
	}
	
	public String printNksRecursively(String prefix) {
		String res = "";
		if (m_isLeaf) {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + ":tk=" + Arrays.toString(this.tk) + " pk="
					+ Arrays.toString(this.pk) + " alpha=" + Utils.doubleToString(alpha, 4) + " c="
					+ Utils.doubleToString(this.getConcentration(), 4) + "\n";
			// res += prefix + ":nk=" + Arrays.toString(this.nk) + "\n";

		} else {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + ":tk=" + Arrays.toString(this.tk) + " pk="
					+ Arrays.toString(this.pk) + " alpha=" + Utils.doubleToString(alpha, 4) + " c="
					+ Utils.doubleToString(this.getConcentration(), 4) + "\n";

			// res += prefix + ":nk=" + Arrays.toString(this.nk) + " alpha="
			// + Utils.doubleToString(alpha, 4) + "\n";
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printNksRecursively(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printNksRecursivelyHGS(String prefix) {
		String res = "";
		if (m_isLeaf) {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + " pk="
					+ Arrays.toString(this.pk) + " alpha=" + Utils.doubleToString(alpha, 4) +"\n";

		} else {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + " pk="
					+ Arrays.toString(this.pk) + " alpha=" + Utils.doubleToString(alpha, 4) + "\n";
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printNksRecursivelyHGS(prefix + " -> " + c);
				}
			}
		}
		return res;
	}

	public String printNksRecursivelyHDP(String prefix) {
		String res = "";
		if (m_isLeaf) {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + ":tk=" + Arrays.toString(this.tk) + " pk="
					+ Arrays.toString(this.pk) + " c="
					+ Utils.doubleToString(this.getConcentration(), 4) + "\n";

		} else {
			res += prefix + ":nk=" + Arrays.toString(this.nk) + ":tk=" + Arrays.toString(this.tk) + " pk="
					+ Arrays.toString(this.pk)  + " c="
					+ Utils.doubleToString(this.getConcentration(), 4) + "\n";
		}

		if (m_sons != null) {
			for (int c = 0; c < m_sons.length; c++) {
				if (m_sons[c] != null) {
					res += m_sons[c].printNksRecursivelyHDP(prefix + " -> " + c);
				}
			}
		}
		return res;
	}
	
	
}
