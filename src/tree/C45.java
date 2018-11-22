package tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import method.HDPMethod;
import method.SmoothingMethod;
import hdp.TyingStrategy;
import hdp.logStirling.LogStirlingFactory;
import hdp.logStirling.LogStirlingGenerator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.mmall.Utils.SUtils;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;

/**
 * <!-- globalinfo-start --> Class for generating a pruned or unpruned C4.5
 * decision tree. For more information, see<br/>
 * <br/>
 * Ross Quinlan (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann
 * Publishers, San Mateo, CA.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;book{Quinlan1993,
 *    address = {San Mateo, CA},
 *    author = {Ross Quinlan},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {C4.5: Programs for Machine Learning},
 *    year = {1993}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -U
 *  Use unpruned tree.
 * </pre>
 * 
 * <pre>
 * -O
 *  Do not collapse tree.
 * </pre>
 * 
 * <pre>
 * -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)
 * </pre>
 * 
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)
 * </pre>
 * 
 * <pre>
 * -R
 *  Use reduced error pruning.
 * </pre>
 * 
 * <pre>
 * -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)
 * </pre>
 * 
 * <pre>
 * -B
 *  Use binary splits only.
 * </pre>
 * 
 * <pre>
 * -S
 *  Don't perform subtree raising.
 * </pre>
 * 
 * <pre>
 * -L
 *  Do not clean up after the tree has been built.
 * </pre>
 * 
 * <pre>
 * -A
 *  Laplace smoothing for predicted probabilities.
 * </pre>
 * 
 * <pre>
 * -J
 *  Do not use MDL correction for info gain on numeric attributes.
 * </pre>
 * 
 * <pre>
 * -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).
 * </pre>
 * 
 * <pre>
 * -doNotMakeSplitPointActualValue
 *  Do not make split point actual value.
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 11194 $
 */
public class C45 extends AbstractClassifier implements OptionHandler, Matchable, Sourcable, WeightedInstancesHandler,
		Summarizable, AdditionalMeasureProducer, PartitionGenerator {

	/** for serialization */
	static final long serialVersionUID = -217733168393644444L;

	private boolean COMPLEXITYPRUNING = false;

	/** The decision tree */
	protected ClassifierTree m_root;

	/** Unpruned tree? */
	protected boolean m_unpruned = false;

	/** Collapse tree? */
	protected boolean m_collapseTree = true;

	/** Confidence level */
	protected float m_CF = 0.25f;

	/** Minimum number of instances */
	protected int m_minNumObj = 2;

	/** Use MDL correction? */
	protected boolean m_useMDLcorrection = true;

	/**
	 * Determines whether probabilities are smoothed using Laplace correction when
	 * predictions are generated
	 */
	protected boolean m_useLaplace = false;

	/** Use reduced error pruning? */
	protected boolean m_reducedErrorPruning = false;

	/** Number of folds for reduced error pruning. */
	protected int m_numFolds = 3;

	/** Binary splits on nominal attributes? */
	protected boolean m_binarySplits = false;

	/** Subtree raising to be performed? */
	protected boolean m_subtreeRaising = true;

	/** Cleanup after the tree has been built. */
	protected boolean m_noCleanup = false;

	/** Random number seed for reduced-error pruning. */
	protected int m_Seed = 1;

	/** Do not relocate split point to actual data value */
	protected boolean m_doNotMakeSplitPointActualValue;

	protected int m_numFoldsPruning = 5;
	/** Training data size. */
	protected double m_SizePer = 1;

	/** hdp parameters. */
	double d = 0.0; // discount
	ArrayList<ConcentrationC45> concentrationsToSample;
	protected TyingStrategy concentrationTyingStrategy = TyingStrategy.LEVEL;
	protected RandomGenerator rng = new MersenneTwister(3071980);
	private int nIterGibbs = 5000;
	private int frequencySamplingC = 1;
	private int nBurnIn = 500;
	private int m_depth;
	ArrayList<ClassifierTree> m_allNodes;
	protected SmoothingMethod method = SmoothingMethod.HGS;
	private HDPMethod methodHDP;
	private int nC;

	/** loocv parameters. */
	protected boolean LOOCV = false;
	ArrayList<ClassifierTree> leaves = new ArrayList<ClassifierTree>();
	ArrayList<ClassifierTree> alphaList = new ArrayList<ClassifierTree>();
	double precision = 0.00001;
	double step = 0.01;
	double lambda = 100000;

	/** recursive LOOCV estimate **/
	protected boolean recursiveLOOCV = false;
	double averagedTrainTime = 0;

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result;

		result = new Capabilities(this);
		result.disableAll();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 * 
	 * @param instances the data to train the classifier with
	 * @throws Exception if classifier can't be built successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		ModelSelection modSelection;

		this.nC = instances.numClasses();

		if (m_binarySplits) {
			modSelection = new BinC45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
					m_doNotMakeSplitPointActualValue);
		} else {
			modSelection = new C45ModelSelection(m_minNumObj, instances, m_useMDLcorrection,
					m_doNotMakeSplitPointActualValue);
		}

		if (!m_reducedErrorPruning) {
			m_root = new C45PruneableClassifierTree(modSelection, !m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup,
					m_collapseTree);
		} else {
			m_root = new PruneableClassifierTree(modSelection, !m_unpruned, m_numFolds, !m_noCleanup, m_Seed);
		}

		if (method == SmoothingMethod.VALIDATE) {
			// use half of the data to learn the structure, use the remaining to learn the
			// parameter

			Random rand = new Random(25011990); // create seeded number generator
			Instances randData = new Instances(instances); // create copy of original data
			randData.randomize(rand);

			Instances trainSet = randData.trainCV(3, 0); // structure learning
			Instances validateSet = randData.testCV(3, 0);// parameter learning
//			System.out.println(randData.numInstances());
//			System.out.println(trainSet.numInstances());
//			System.out.println(validateSet.numInstances());
			trainSet.setClassIndex(trainSet.numAttributes() - 1);
			validateSet.setClassIndex(validateSet.numAttributes() - 1);

			m_root.buildClassifier(trainSet);
//			learnStructure(trainSet);
			m_root.clearCounts();
			m_root.learnParameter(validateSet);
			convertCountToProbs("None");
			return;
		} else {
			m_root.buildClassifier(instances);
		}

		switch (method) {
		case None:
			this.convertCountToProbs(method.toString());// no smoothing
			break;
		case LAPLACE:
			this.m_useLaplace = true;
			this.convertCountToProbs(method.toString());
			break;
		case M_estimation:
			this.convertCountToProbs(method.toString());
			break;
		case HDP:

			if (methodHDP.equals(HDPMethod.Alpha)) {
				// initialize HDP using recursive LOOCV.
				this.recursiveLOOCV = true;
				treeTraversal(m_root);
//				 this.recursiveLOOCVCost();
				this.stepGradient();
//				 this.printTree();
				this.concentrationTyingStrategy = TyingStrategy.SAME_PARENT;
			}

			LogStirlingGenerator lgcache = LogStirlingFactory.newLogStirlingGenerator(instances.numInstances(), 0.0);
			
//			LogStirlingCache lgcache = new LogStirlingCache(0.0,instances.numInstances());
			m_root.setLogStirlingCache(lgcache);
			smooth();
//			this.printTree();
			break;
		case HGS:
			treeTraversal(m_root);
			stepGradient();
			double[] sumalpha = new double[this.nC];
			calculatePkForLeaves(m_root, 0, sumalpha, false);
//			this.printTree();
			break;
		case RECURSIVE:
			this.recursiveLOOCV = true;
			treeTraversal(m_root);
			// this.recursiveLOOCVCost();
			this.stepGradient();
//			 this.printTree();
			this.calculatePkForLeavesRecusive();
			// this.printTree();
			break;
		case MBranch:
			MBranchSmoothing();
			break;
		case OptiMestimation:
			double bestM = optimizedMestimation(instances);
			SUtils.m_MEsti = bestM;
			System.out.println("Selecting smoothing param=" + SUtils.getMEsti());
			convertCountToProbs("M_estimation");
			break;
		default:
			break;
		}

		if (m_binarySplits) {
			((BinC45ModelSelection) modSelection).cleanup();
		} else {
			((C45ModelSelection) modSelection).cleanup();
		}
	}

	/**
	 * Returns class probabilities for an instance.
	 * 
	 * @param instance the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception if distribution can't be computed successfully
	 */
	@Override
	public final double[] distributionForInstance(Instance instance) throws Exception {
		return m_root.distributionForInstance(instance);
	}

	private double optimizedMestimation(Instances data) {

		Random rand = new Random(25011990); // create seeded number generator
		C45 classifier = new C45();
//		classifier.setReducedErrorPruning(true);
		// tree.setComplexityPruning(true);
		classifier.setUnpruned(m_unpruned);
		classifier.setMethod(SmoothingMethod.M_estimation);

		double bestM = 1.0;
		double bestRMSE = Double.MAX_VALUE;

		double[] mValues = { 0, 0.05, 0.2, 1, 5, 20 };
		for (double m : mValues) {
			Evaluation evall;
			SUtils.setMEsti(m);
			System.out.println(SUtils.getMEsti());
			try {
				evall = new Evaluation(data);
				evall.crossValidateModel(classifier, data, 3, rand);

				double rmse = evall.rootMeanSquaredError();
				if (bestRMSE > rmse) {
					bestRMSE = rmse;
					bestM = m;
				}
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return bestM;
	}

	private void MBranchSmoothing() {
		// TODO Auto-generated method stub

	}

	private void calulatePartialDerivativeDownUp() {
		this.m_root.calculatePartialDerivativeDownUp();
	}

	private void calculateLOOEstimatesTopDown() {

		double[] uniform = new double[nC];
		for (int i = 0; i < nC; i++) {
			uniform[i] = (double) 1 / nC;
		}
		// the base distribution for root is uniform distribution
		this.m_root.calculateLOOestimatesTopDown(uniform, 1.0);
	}

	private void calculatePkForLeavesRecusive() {
		double[] uniform = new double[nC];
		for (int i = 0; i < nC; i++) {
			uniform[i] = (double) 1 / nC;
		}
		// the base distribution for root is uniform distribution
		this.m_root.calculatePKforLeavesTopDown(uniform, 1.0);
	}

	private void convertCountToProbs(String s) {
		this.m_root.convertCountToProbs(s);
	}

	private boolean getComplexityPruning() {
		return this.COMPLEXITYPRUNING;
	}

	protected void unprune() {
		this.m_root.unprune();
	}

	/**
	 * Method for performing one fold in the cross-validation of minimal
	 * cost-complexity pruning. Generates a sequence of alpha-values with error
	 * estimates for the corresponding (partially pruned) trees, given the test set
	 * of that fold.
	 * 
	 * @param alphas array to hold the generated alpha-values
	 * @param errors array to hold the corresponding error estimates
	 * @param test   test set of that fold (to obtain error estimates)
	 * @return the iteration of the pruning
	 * @throws Exception if something goes wrong
	 */
	public int prune(double[] alphas, double[] errors, Instances test) throws Exception {

		Vector<ClassifierTree> nodeList;

		// determine training error of subtrees (both with and without replacing
		// a
		// subtree),
		// and calculate alpha-values from them
		modelErrors();
		treeErrors();
		calculateAlphas();

		// get list of all inner nodes in the tree
		nodeList = getInnerNodes();

		boolean prune = (nodeList.size() > 0);

		// alpha_0 is always zero (unpruned tree)
		alphas[0] = 0;

		Evaluation eval;

		// error of unpruned tree
		if (errors != null) {
			eval = new Evaluation(test);
			eval.evaluateModel(this, test);
			errors[0] = eval.errorRate();
		}

		int iteration = 0;
		double preAlpha = Double.MAX_VALUE;
		while (prune) {

			iteration++;

			// get node with minimum alpha
			ClassifierTree nodeToPrune = nodeToPrune(nodeList);

			// do not set m_sons null, want to unprune
			nodeToPrune.m_isLeaf = true;

			// normally would not happen
			if (nodeToPrune.m_Alpha == preAlpha) {
				iteration--;
				treeErrors();
				calculateAlphas();
				nodeList = getInnerNodes();
				prune = (nodeList.size() > 0);
				continue;
			}

			// get alpha-value of node
			alphas[iteration] = nodeToPrune.m_Alpha;

			// log error
			if (errors != null) {
				eval = new Evaluation(test);
				eval.evaluateModel(this, test);
				errors[iteration] = eval.errorRate();
			}
			preAlpha = nodeToPrune.m_Alpha;

			// update errors/alphas
			treeErrors();
			calculateAlphas();

			nodeList = getInnerNodes();
			prune = (nodeList.size() > 0);
		}

		// set last alpha 1 to indicate end
		alphas[iteration + 1] = 1.0;
		return iteration;
	}

	private Vector<ClassifierTree> getInnerNodes() {
		return m_root.getInnerNodes();
	}

	/**
	 * Method to count the number of inner nodes in the tree.
	 * 
	 * @return the number of inner nodes
	 */
	public int numInnerNodes() {
		return m_root.numInnerNodes();
	}

	public void modelErrors() throws Exception {
		this.m_root.modelErrors();
	}

	/**
	 * Compute sorted indices, weights and class probabilities for a given dataset.
	 * Return total weights of the data at the node.
	 * 
	 * @param data          training data
	 * @param sortedIndices sorted indices of instances at the node
	 * @param weights       weights of instances at the node
	 * @param classProbs    class probabilities at the node
	 * @return total weights of instances at the node
	 * @throws Exception if something goes wrong
	 */
	protected double computeSortedInfo(Instances data, int[][] sortedIndices, double[][] weights, double[] classProbs)
			throws Exception {

		// Create array of sorted indices and weights
		double[] vals = new double[data.numInstances()];
		for (int j = 0; j < data.numAttributes(); j++) {
			if (j == data.classIndex()) {
				continue;
			}
			weights[j] = new double[data.numInstances()];

			if (data.attribute(j).isNominal()) {

				// Handling nominal attributes. Putting indices of
				// instances with missing values at the end.
				sortedIndices[j] = new int[data.numInstances()];
				int count = 0;
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (!inst.isMissing(j)) {
						sortedIndices[j][count] = i;
						weights[j][count] = inst.weight();
						count++;
					}
				}
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					if (inst.isMissing(j)) {
						sortedIndices[j][count] = i;
						weights[j][count] = inst.weight();
						count++;
					}
				}
			} else {

				// Sorted indices are computed for numeric attributes
				// missing values instances are put to end
				for (int i = 0; i < data.numInstances(); i++) {
					Instance inst = data.instance(i);
					vals[i] = inst.value(j);
				}
				sortedIndices[j] = Utils.sort(vals);
				for (int i = 0; i < data.numInstances(); i++) {
					weights[j][i] = data.instance(sortedIndices[j][i]).weight();
				}
			}
		}

		// Compute initial class counts
		double totalWeight = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.instance(i);
			classProbs[(int) inst.classValue()] += inst.weight();
			totalWeight += inst.weight();
		}

		return totalWeight;
	}

	protected ClassifierTree nodeToPrune(Vector<ClassifierTree> nodeList) {
		if (nodeList.size() == 0) {
			return null;
		}
		if (nodeList.size() == 1) {
			return nodeList.elementAt(0);
		}
		ClassifierTree returnNode = nodeList.elementAt(0);
		double baseAlpha = returnNode.m_Alpha;
		for (int i = 1; i < nodeList.size(); i++) {
			ClassifierTree node = nodeList.elementAt(i);
			if (node.m_Alpha < baseAlpha) {
				baseAlpha = node.m_Alpha;
				returnNode = node;
			} else if (node.m_Alpha == baseAlpha) { // break tie
				if (node.numLeaves() > returnNode.numLeaves()) {
					returnNode = node;
				}
			}
		}
		return returnNode;
	}

	/**
	 * Classifies an instance.
	 * 
	 * @param instance the instance to classify
	 * @return the classification for the instance
	 * @throws Exception if instance can't be classified successfully
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {

		return m_root.classifyInstance(instance);
	}

	/**
	 * Returns tree in prefix order.
	 * 
	 * @return the tree in prefix order
	 * @throws Exception if something goes wrong
	 */
	@Override
	public String prefix() throws Exception {

		return m_root.prefix();
	}

	/**
	 * Returns tree as an if-then statement.
	 * 
	 * @param className the name of the Java class
	 * @return the tree as a Java if-then type statement
	 * @throws Exception if something goes wrong
	 */
	@Override
	public String toSource(String className) throws Exception {

		StringBuffer[] source = m_root.toSource(className);
		return "class " + className + " {\n\n" + "  public static double classify(Object[] i)\n"
				+ "    throws Exception {\n\n" + "    double p = Double.NaN;\n" + source[0] // Assignment
																							// code
				+ "    return p;\n" + "  }\n" + source[1] // Support code
				+ "}\n";
	}

	/**
	 * Returns an enumeration describing the available options.
	 * 
	 * Valid options are:
	 * <p>
	 * 
	 * -U <br>
	 * Use unpruned tree.
	 * <p>
	 * 
	 * -C confidence <br>
	 * Set confidence threshold for pruning. (Default: 0.25)
	 * <p>
	 * 
	 * -M number <br>
	 * Set minimum number of instances per leaf. (Default: 2)
	 * <p>
	 * 
	 * -R <br>
	 * Use reduced error pruning. No subtree raising is performed.
	 * <p>
	 * 
	 * -N number <br>
	 * Set number of folds for reduced error pruning. One fold is used as the
	 * pruning set. (Default: 3)
	 * <p>
	 * 
	 * -B <br>
	 * Use binary splits for nominal attributes.
	 * <p>
	 * 
	 * -S <br>
	 * Don't perform subtree raising.
	 * <p>
	 * 
	 * -L <br>
	 * Do not clean up after the tree has been built.
	 * 
	 * -A <br>
	 * If set, Laplace smoothing is used for predicted probabilites.
	 * <p>
	 * 
	 * -Q <br>
	 * The seed for reduced-error pruning.
	 * <p>
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(13);

		newVector.addElement(new Option("\tUse unpruned tree.", "U", 0, "-U"));
		newVector.addElement(new Option("\tDo not collapse tree.", "O", 0, "-O"));
		newVector.addElement(new Option("\tSet confidence threshold for pruning.\n" + "\t(default 0.25)", "C", 1,
				"-C <pruning confidence>"));
		newVector.addElement(new Option("\tSet minimum number of instances per leaf.\n" + "\t(default 2)", "M", 1,
				"-M <minimum number of instances>"));
		newVector.addElement(new Option("\tUse reduced error pruning.", "R", 0, "-R"));
		newVector.addElement(new Option("\tSet number of folds for reduced error\n"
				+ "\tpruning. One fold is used as pruning set.\n" + "\t(default 3)", "N", 1, "-N <number of folds>"));
		newVector.addElement(new Option("\tUse binary splits only.", "B", 0, "-B"));
		newVector.addElement(new Option("\tDo not perform subtree raising.", "S", 0, "-S"));
		newVector.addElement(new Option("\tDo not clean up after the tree has been built.", "L", 0, "-L"));
		newVector.addElement(new Option("\tLaplace smoothing for predicted probabilities.", "A", 0, "-A"));
		newVector.addElement(
				new Option("\tDo not use MDL correction for info gain on numeric attributes.", "J", 0, "-J"));
		newVector.addElement(new Option("\tSeed for random data shuffling (default 1).", "Q", 1, "-Q <seed>"));
		newVector.addElement(new Option("\tDo not make split point actual value.", "-doNotMakeSplitPointActualValue", 0,
				"-doNotMakeSplitPointActualValue"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * 
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 * -U
	 *  Use unpruned tree.
	 * </pre>
	 * 
	 * <pre>
	 * -O
	 *  Do not collapse tree.
	 * </pre>
	 * 
	 * <pre>
	 * -C &lt;pruning confidence&gt;
	 *  Set confidence threshold for pruning.
	 *  (default 0.25)
	 * </pre>
	 * 
	 * <pre>
	 * -M &lt;minimum number of instances&gt;
	 *  Set minimum number of instances per leaf.
	 *  (default 2)
	 * </pre>
	 * 
	 * <pre>
	 * -R
	 *  Use reduced error pruning.
	 * </pre>
	 * 
	 * <pre>
	 * -N &lt;number of folds&gt;
	 *  Set number of folds for reduced error
	 *  pruning. One fold is used as pruning set.
	 *  (default 3)
	 * </pre>
	 * 
	 * <pre>
	 * -B
	 *  Use binary splits only.
	 * </pre>
	 * 
	 * <pre>
	 * -S
	 *  Don't perform subtree raising.
	 * </pre>
	 * 
	 * <pre>
	 * -L
	 *  Do not clean up after the tree has been built.
	 * </pre>
	 * 
	 * <pre>
	 * -A
	 *  Laplace smoothing for predicted probabilities.
	 * </pre>
	 * 
	 * <pre>
	 * -J
	 *  Do not use MDL correction for info gain on numeric attributes.
	 * </pre>
	 * 
	 * <pre>
	 * -Q &lt;seed&gt;
	 *  Seed for random data shuffling (default 1).
	 * </pre>
	 * 
	 * <pre>
	 * -doNotMakeSplitPointActualValue
	 *  Do not make split point actual value.
	 * </pre>
	 * 
	 * <!-- options-end -->
	 * 
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		// Other options
		String minNumString = Utils.getOption('M', options);
		if (minNumString.length() != 0) {
			m_minNumObj = Integer.parseInt(minNumString);
		} else {
			m_minNumObj = 2;
		}
		m_binarySplits = Utils.getFlag('B', options);
		m_useLaplace = Utils.getFlag('A', options);
		m_useMDLcorrection = !Utils.getFlag('J', options);

		// Pruning options
		m_unpruned = Utils.getFlag('U', options);
		m_collapseTree = !Utils.getFlag('O', options);
		m_subtreeRaising = !Utils.getFlag('S', options);
		m_noCleanup = Utils.getFlag('L', options);
		m_doNotMakeSplitPointActualValue = Utils.getFlag("doNotMakeSplitPointActualValue", options);
		if ((m_unpruned) && (!m_subtreeRaising)) {
			throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
		}
		m_reducedErrorPruning = Utils.getFlag('R', options);
		if ((m_unpruned) && (m_reducedErrorPruning)) {
			throw new Exception("Unpruned tree and reduced error pruning can't be selected " + "simultaneously!");
		}
		String confidenceString = Utils.getOption('C', options);
		if (confidenceString.length() != 0) {
			if (m_reducedErrorPruning) {
				throw new Exception("Setting the confidence doesn't make sense " + "for reduced error pruning.");
			} else if (m_unpruned) {
				throw new Exception("Doesn't make sense to change confidence for unpruned " + "tree!");
			} else {
				m_CF = (new Float(confidenceString)).floatValue();
				if ((m_CF <= 0) || (m_CF >= 1)) {
					throw new Exception("Confidence has to be greater than zero and smaller " + "than one!");
				}
			}
		} else {
			m_CF = 0.25f;
		}
		String numFoldsString = Utils.getOption('N', options);
		if (numFoldsString.length() != 0) {
			if (!m_reducedErrorPruning) {
				throw new Exception("Setting the number of folds" + " doesn't make sense if"
						+ " reduced error pruning is not selected.");
			} else {
				m_numFolds = Integer.parseInt(numFoldsString);
			}
		} else {
			m_numFolds = 3;
		}
		String seedString = Utils.getOption('Q', options);
		if (seedString.length() != 0) {
			m_Seed = Integer.parseInt(seedString);
		} else {
			m_Seed = 1;
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 * 
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		if (m_noCleanup) {
			options.add("-L");
		}
		if (!m_collapseTree) {
			options.add("-O");
		}
		if (m_unpruned) {
			options.add("-U");
		} else {
			if (!m_subtreeRaising) {
				options.add("-S");
			}
			if (m_reducedErrorPruning) {
				options.add("-R");
				options.add("-N");
				options.add("" + m_numFolds);
				options.add("-Q");
				options.add("" + m_Seed);
			} else {
				options.add("-C");
				options.add("" + m_CF);
			}
		}
		if (m_binarySplits) {
			options.add("-B");
		}
		options.add("-M");
		options.add("" + m_minNumObj);
		if (m_useLaplace) {
			options.add("-A");
		}
		if (!m_useMDLcorrection) {
			options.add("-J");
		}
		if (m_doNotMakeSplitPointActualValue) {
			options.add("-doNotMakeSplitPointActualValue");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	public int getSeed() {

		return m_Seed;
	}

	public void setSeed(int newSeed) {

		m_Seed = newSeed;
	}

	public boolean getUseLaplace() {

		return m_useLaplace;
	}

	public boolean getUseMDLcorrection() {

		return m_useMDLcorrection;
	}

	public void setUseMDLcorrection(boolean newuseMDLcorrection) {

		m_useMDLcorrection = newuseMDLcorrection;
	}

	@Override
	public String toString() {

		if (m_root == null) {
			return "No classifier built";
		}
		if (m_unpruned) {
			return "J48 unpruned tree\n------------------\n" + m_root.toString();
		} else {
			return "J48 pruned tree\n------------------\n" + m_root.toString();
		}
	}

	@Override
	public String toSummaryString() {

		return "Number of leaves: " + m_root.numLeaves() + "\n" + "Size of the tree: " + m_root.numNodes() + "\n";
	}

	public double measureTreeSize() {
		return m_root.numNodes();
	}

	public double measureNumLeaves() {
		return m_root.numLeaves();
	}

	public double measureNumRules() {
		return m_root.numLeaves();
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * 
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(3);
		newVector.addElement("measureTreeSize");
		newVector.addElement("measureNumLeaves");
		newVector.addElement("measureNumRules");
		return newVector.elements();
	}

	/**
	 * Returns the value of the named measure
	 * 
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	@Override
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
			return measureNumRules();
		} else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
			return measureTreeSize();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
			return measureNumLeaves();
		} else {
			throw new IllegalArgumentException(additionalMeasureName + " not supported (j48)");
		}
	}

	public boolean getUnpruned() {

		return m_unpruned;
	}

	public void setUnpruned(boolean v) {

		if (v) {
			m_reducedErrorPruning = false;
		}
		m_unpruned = v;
	}

	public String collapseTreeTipText() {
		return "Whether parts are removed that do not reduce training error.";
	}

	public boolean getCollapseTree() {

		return m_collapseTree;
	}

	public void setCollapseTree(boolean v) {

		m_collapseTree = v;
	}

	public String confidenceFactorTipText() {
		return "The confidence factor used for pruning (smaller values incur " + "more pruning).";
	}

	public float getConfidenceFactor() {

		return m_CF;
	}

	public void setConfidenceFactor(float v) {

		m_CF = v;
	}

	public String minNumObjTipText() {
		return "The minimum number of instances per leaf.";
	}

	public int getMinNumObj() {

		return m_minNumObj;
	}

	public void setMinNumObj(int v) {

		m_minNumObj = v;
	}

	public String reducedErrorPruningTipText() {
		return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
	}

	public boolean getReducedErrorPruning() {

		return m_reducedErrorPruning;
	}

	public void setReducedErrorPruning(boolean v) {

		if (v) {
			m_unpruned = false;
		}
		m_reducedErrorPruning = v;
	}

	public String numFoldsTipText() {
		return "Determines the amount of data used for reduced-error pruning. "
				+ " One fold is used for pruning, the rest for growing the tree.";
	}

	public int getNumFolds() {

		return m_numFolds;
	}

	public void setNumFolds(int v) {

		m_numFolds = v;
	}

	public String binarySplitsTipText() {
		return "Whether to use binary splits on nominal attributes when " + "building the trees.";
	}

	public boolean getBinarySplits() {

		return m_binarySplits;
	}

	public void setBinarySplits(boolean v) {

		m_binarySplits = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String subtreeRaisingTipText() {
		return "Whether to consider the subtree raising operation when pruning.";
	}

	/**
	 * Get the value of subtreeRaising.
	 * 
	 * @return Value of subtreeRaising.
	 */
	public boolean getSubtreeRaising() {

		return m_subtreeRaising;
	}

	/**
	 * Set the value of subtreeRaising.
	 * 
	 * @param v Value to assign to subtreeRaising.
	 */
	public void setSubtreeRaising(boolean v) {

		m_subtreeRaising = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String saveInstanceDataTipText() {
		return "Whether to save the training data for visualization.";
	}

	/**
	 * Check whether instance data is to be saved.
	 * 
	 * @return true if instance data is saved
	 */
	public boolean getSaveInstanceData() {

		return m_noCleanup;
	}

	/**
	 * Set whether instance data is to be saved.
	 * 
	 * @param v true if instance data is to be saved
	 */
	public void setSaveInstanceData(boolean v) {

		m_noCleanup = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String doNotMakeSplitPointActualValueTipText() {
		return "If true, the split point is not relocated to an actual data value."
				+ " This can yield substantial speed-ups for large datasets with numeric attributes.";
	}

	/**
	 * Gets the value of doNotMakeSplitPointActualValue.
	 * 
	 * @return the value
	 */
	public boolean getDoNotMakeSplitPointActualValue() {
		return m_doNotMakeSplitPointActualValue;
	}

	/**
	 * Sets the value of doNotMakeSplitPointActualValue.
	 * 
	 * @param m_doNotMakeSplitPointActualValue the value to set
	 */
	public void setDoNotMakeSplitPointActualValue(boolean m_doNotMakeSplitPointActualValue) {
		this.m_doNotMakeSplitPointActualValue = m_doNotMakeSplitPointActualValue;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 11194 $");
	}

	/**
	 * Builds the classifier to generate a partition.
	 */
	@Override
	public void generatePartition(Instances data) throws Exception {

		buildClassifier(data);
	}

	/**
	 * Computes an array that indicates node membership.
	 */
	@Override
	public double[] getMembershipValues(Instance inst) throws Exception {

		return m_root.getMembershipValues(inst);
	}

	@Override
	public int numElements() throws Exception {

		return m_root.numNodes();
	}

	/**
	 * Runs the Gibbs sampling for the whole tree with given discount and
	 * concentration parameters
	 * 
	 * @return the log likelihood of the optimized tree
	 */
	public double smooth() {

		m_depth = getTreeDepth(m_root);

		// Creating and tying concentrations
		concentrationsToSample = new ArrayList<>();
		switch (concentrationTyingStrategy) {
		case NONE:
			for (int depth = m_depth; depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ClassifierTree> nodes = getAllNodesAtDepth(depth);
				for (ClassifierTree node : nodes) {
					ConcentrationC45 c = new ConcentrationC45(this.nC);
					concentrationsToSample.add(c);
					node.c = c;
					c.addNode(node);
				}
			}
			break;
		case SAME_PARENT:
			for (int depth = m_depth - 1; depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ClassifierTree> nodes = getAllNodesAtDepth(depth);
				for (ClassifierTree parent : nodes) {

					if (parent.m_sons != null) {
						// creating concentration
						ConcentrationC45 c = new ConcentrationC45(this.nC);
						concentrationsToSample.add(c);
						for (int j = 0; j < parent.m_sons.length; j++) {
							ClassifierTree child = parent.m_sons[j];
							if (child != null) {
								child.c = c;
								c.addNode(child);
							}
						}
					}
				}
			}
			break;
		case LEVEL:
			for (int depth = m_depth; depth >= 0; depth--) {
				// tying all children of a node
				ArrayList<ClassifierTree> nodes = getAllNodesAtDepth(depth);
				ConcentrationC45 c = new ConcentrationC45(this.nC);
				concentrationsToSample.add(c);
				for (ClassifierTree node : nodes) {
					node.c = c;
					c.addNode(node);
				}
			}
			break;
		case SINGLE:
			ConcentrationC45 c = new ConcentrationC45(this.nC);
			concentrationsToSample.add(c);
			for (int depth = m_depth; depth > 0; depth--) {
				// tying all children of a node
				ArrayList<ClassifierTree> nodes = getAllNodesAtDepth(depth);
				for (ClassifierTree node : nodes) {
					node.c = c;
					c.addNode(node);
				}
			}
			break;
		default:
			break;
		}

//		this.m_root.c = new ConcentrationC45(this.nC);
		this.m_root.c = new ConcentrationC45();

		// initialize concentration using alphas
		if (methodHDP.equals(HDPMethod.Alpha)) {
			this.convertAlphaToConcentratoins();
			this.printTree("HDP");
			this.methodHDP = HDPMethod.Expected;
		}

		m_root.prepareForSamplingTk();

		// Gibbs sampling of the tks, c
		for (int iter = 0; iter < nIterGibbs; iter++) {
			// sample tks once
			for (int depth = m_depth; depth >= 0; depth--) {
				ArrayList<ClassifierTree> nodes = getAllNodesAtDepth(depth);
				for (ClassifierTree node : nodes) {
					node.sampleTks();
				}
			}

			// sample c
			if ((iter + frequencySamplingC / 2) % frequencySamplingC == 0) {
				// sample c once
				for (ConcentrationC45 c : concentrationsToSample) {
					c.sample(rng);
				}
			}

			if (iter >= nBurnIn) {
				this.recordAndAverageProbabilities();
			}
		}

		double score = logScoreTree();
		return score;
	}

	/** pkAveraged **/
	private void recordAndAverageProbabilities() {
		m_root.computeProbabilities();
		m_root.recordAndAverageProbabilities();
	}

	private void convertAlphaToConcentratoins() {
		this.m_root.convertAlphaToConcentrations();
	}

	protected int getTreeDepth(ClassifierTree node) {
		if (node != null) {
			if (node.m_sons != null) {
				int[] depth = new int[node.m_sons.length];
				for (int i = 0; i < node.m_sons.length; i++) {
					depth[i] = getTreeDepth(node.m_sons[i]);
				}
				return 1 + depth[Utils.maxIndex(depth)];
			}
			return 0;
		} else
			return 0;
	}

	protected double logScoreTree() {
		return m_root.logScoreSubTree();
	}

	private ArrayList<ClassifierTree> getAllNodesAtDepth(int depth) {
		return m_root.getAllNodesAtRelativeDepth(depth);
	}

	private void printTree(String s) {
		if (s.equalsIgnoreCase("HGS")) {
			System.out.println(m_root.printNksRecursivelyHGS("root"));
		} else if (s.equalsIgnoreCase("HDP")) {
			System.out.println(m_root.printNksRecursivelyHDP("root"));
		}

	}

	public void setNumFoldsPruning(int value) {
		m_numFoldsPruning = value;
	}

	public void treeErrors() throws Exception {
		this.m_root.treeErrors();
	}

	public void calculateAlphas() throws Exception {
		this.m_root.calculateAlphas();
	}

	public void prune(double alpha) throws Exception {

		Vector<ClassifierTree> nodeList;

		// determine training error of pruned subtrees (both with and without
		// replacing a subtree),
		// and calculate alpha-values from them
		modelErrors();
		treeErrors();
		calculateAlphas();

		// get list of all inner nodes in the tree
		nodeList = getInnerNodes();

		boolean prune = (nodeList.size() > 0);
		double preAlpha = Double.MAX_VALUE;
		while (prune) {

			// select node with minimum alpha
			ClassifierTree nodeToPrune = nodeToPrune(nodeList);

			// want to prune if its alpha is smaller than alpha
			if (nodeToPrune.m_Alpha > alpha) {
				break;
			}

			nodeToPrune.makeLeaf();

			// normally would not happen
			if (nodeToPrune.m_Alpha == preAlpha) {
				nodeToPrune.makeLeaf();
				treeErrors();
				calculateAlphas();
				nodeList = getInnerNodes();
				prune = (nodeList.size() > 0);
				continue;
			}
			preAlpha = nodeToPrune.m_Alpha;

			// update tree errors and alphas
			treeErrors();
			calculateAlphas();

			nodeList = getInnerNodes();
			prune = (nodeList.size() > 0);
		}
	}

	private void setComplexityPruning(boolean b) {
		this.COMPLEXITYPRUNING = b;
	}

	/**
	 * LOOCV
	 */

	/**
	 * return all the internal nodes, all the leaves, and leaves under each internal
	 * nodes
	 */
	void treeTraversal(ClassifierTree node) {

		if (node.m_isLeaf) {
			leaves.add(node);
			return;
		}
		// internal node
		alphaList.add(node);
		node.leavesUnderThisNode = new ArrayList<ClassifierTree>();

		// loocv probability for internal node
		for (int i = 0; i < this.nC; i++) {
			if (node.nk[i] > 0) {
				node.pk[i] = (double) (node.nk[i] - 1) / (Utils.sum(node.nk) - 1);
				// node.pk[i] = SUtils.MEsti((node.nk[i] - 1),
				// Utils.sum(node.nk),
				// nC);
			}
		}

		if (node.m_sons != null) {
			for (ClassifierTree son : node.m_sons) {
				if (son != null)
					treeTraversal(son);
				if (son.m_isLeaf)
					node.leavesUnderThisNode.add(son);
				else {
					node.leavesUnderThisNode.addAll(son.leavesUnderThisNode);
				}
			}
		}
	}

	void stepGradient() {
		double currentCost = LOOCVCost(); // alphas are all initialized as 1
		System.out.println(0 + "\t" + currentCost + "\t");
		double costDifference = currentCost;
		int iter = 0;
		double newCost = 0;
		double minimumCost = Double.MAX_VALUE;
		ArrayList<Double> bestAlphas = new ArrayList<Double>();
		while (costDifference > this.precision) {
//		while (iter < 2000) {
			ClassifierTree node;
			double tempSum = 0;
			boolean negative = false;
			for (int i = 0; i < this.alphaList.size(); i++) {
				node = alphaList.get(i);
				node.alpha -= step * node.partialDerivative;
				if (node.alpha < 0) {
					negative = true;
				}
				// if any alpha becomes negative, retore to the previous one, finish gradient
//				if (node.alpha < 0) {
//					for (int j = 0; j <= i; j++) {
//						ClassifierTree temp = alphaList.get(j);
//						temp.alpha += step * temp.partialDerivative;
//					}
//					for (int c = 0; c < this.alphaList.size(); i++) {
//						System.out.println(alphaList.get(c).alpha);
//					}
//					break;
//				}
			}
			if (negative) {
				// add L2 norm to the cost, change the partial deriviative
				for (int i = 0; i < this.alphaList.size(); i++) {
					node = alphaList.get(i);
					System.out.print(node.alpha+"\t");
					 node.alpha += step * node.partialDerivative;
//					if (node.alpha < 0) {
//						tempSum += lambda * Math.pow(node.alpha, 2);
//					}
					 System.out.println(node.alpha);
				}
			}

			newCost = this.LOOCVCost() + tempSum;
			costDifference = currentCost - newCost;

//			for (int i = 0; i < this.alphaList.size(); i++) {
////		alphaList.get(i).alpha = bestAlphas.get(i);
//				System.out.println(alphaList.get(i).alpha);
//			}
			currentCost = newCost;
//			if (minimumCost > currentCost) {
//				minimumCost = currentCost;
//				bestAlphas = new ArrayList<Double>();
//				for (int i = 0; i < this.alphaList.size(); i++) {
//					bestAlphas.add(alphaList.get(i).alpha);
//				}
//			}
			iter++;
			System.out.println(iter + "\t" + newCost + "\t");
		}

//		System.out.println(minimumCost);
//		System.out.println("gradient descent finished.");
//		for (int i = 0; i < bestAlphas.size(); i++) {
//			System.out.println(bestAlphas.get(i));
//		}
//		System.out.println();

//		for (int i = 0; i < this.alphaList.size(); i++) {
////			alphaList.get(i).alpha = bestAlphas.get(i);
//			System.out.println(alphaList.get(i).alpha);
//		}
//		this.printTree("HGS");
		System.out.println();
	}

	/**
	 * Minimize the cost function
	 */
	private double LOOCVCost() {

		// get probability of all the nodes
		if (this.recursiveLOOCV) {
			calculateLOOEstimatesTopDown();
			calulatePartialDerivativeDownUp();
		} else {
			calculatePkForLeaves(m_root, 0, new double[nC], true);
		}

		double loocvCost = 0;
		for (int c = 0; c < this.leaves.size(); c++) {
			ClassifierTree node = leaves.get(c);

			for (int k = 0; k < this.nC; k++) {
				loocvCost += node.nk[k] * Math.pow(1 - node.pk[k], 2);
			}
		}

		return loocvCost;
	}

	public void calculatePkForLeaves(ClassifierTree node, double sumalpha, double[] sumpk, boolean isGD) {
		if (node.m_isLeaf) {

			if (isGD) {
				node.alc = new double[this.nC];
				for (int i = 0; i < this.nC; i++) {
					if (node.nk[i] > 0) {
						node.pk[i] = (node.nk[i] - 1 + sumpk[i]) / (Utils.sum(node.nk) - 1 + sumalpha);
						node.alc[i] = node.nk[i] * (1 - node.pk[i]) / (Utils.sum(node.nk) - 1 + sumalpha);
					}
				}
			} else {
				node.pkAveraged = new double[this.nC];
				for (int i = 0; i < this.nC; i++) {
					node.pkAveraged[i] = (node.nk[i] + sumpk[i]) / (Utils.sum(node.nk) + sumalpha);
				}
			}

			return;
		}

		sumalpha += node.alpha;
		int sum = Utils.sum(node.nk);
		if (isGD) {

			for (int i = 0; i < this.nC; i++) {
				sumpk[i] += node.pk[i] * node.alpha;
			}
		} else {

			for (int i = 0; i < this.nC; i++) {
				// node.pk[i] = SUtils.MEsti(node.nk[i], sum, this.nC);
				node.pk[i] = (double) node.nk[i] / sum;
				// node.pk[i] = (double) (node.nk[i]+1)/(sum+this.nC);
			}

			SUtils.normalize(node.pk);
			//
			for (int i = 0; i < this.nC; i++) {
				sumpk[i] += node.pk[i] * node.alpha;
			}
		}

		// first recur on left subtree
		if (node.m_sons != null) {
			for (int s = 0; s < node.m_sons.length; s++) {
				calculatePkForLeaves(node.m_sons[s], sumalpha, sumpk, isGD);
			}
			sumalpha -= node.alpha;
			for (int i = 0; i < this.nC; i++) {
				sumpk[i] -= node.pk[i] * node.alpha;
			}
		}

		// now deal with the node
		if (isGD) {
			if (node.leavesUnderThisNode != null) {
				node.partialDerivative = 0;
				for (int c = 0; c < node.leavesUnderThisNode.size(); c++) {
					ClassifierTree son = node.leavesUnderThisNode.get(c);
					for (int k = 0; k < this.nC; k++) {
						node.partialDerivative += 2 * son.alc[k] * (son.pk[k] - node.pk[k]);
					}
				}

				if (node.alpha < 0) {
					node.partialDerivative += 2 * lambda * node.alpha;
				}
			}
		}
	}

	public void setMethod(SmoothingMethod method2) {
		this.method = method2;
	}

	public void setTyingStrategy(String m_Tying) {

		if (m_Tying.equalsIgnoreCase("LEVEL"))
			this.concentrationTyingStrategy = TyingStrategy.LEVEL;
		else if (m_Tying.equalsIgnoreCase("SAME_PARENT"))
			this.concentrationTyingStrategy = TyingStrategy.SAME_PARENT;
		else if (m_Tying.equalsIgnoreCase("SINGLE"))
			this.concentrationTyingStrategy = TyingStrategy.SINGLE;
		else if (m_Tying.equalsIgnoreCase("NONE"))
			this.concentrationTyingStrategy = TyingStrategy.NONE;
	}

	public void setGibbsIteration(int gibbs) {

		this.nIterGibbs = gibbs;
	}

	public void setHDPMethod(HDPMethod method) {
		this.methodHDP = method;

	}
}
