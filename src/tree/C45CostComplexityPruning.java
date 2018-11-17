package tree;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import hdp.Concentration;
import hdp.ProbabilityNode;
import hdp.TyingStrategy;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.classifiers.Evaluation;
import weka.classifiers.mmall.Utils.SUtils;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
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
public class C45CostComplexityPruning extends AbstractClassifier
		implements OptionHandler, Drawable, Matchable, Sourcable, WeightedInstancesHandler, Summarizable,
		AdditionalMeasureProducer, TechnicalInformationHandler, PartitionGenerator {

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
	 * Determines whether probabilities are smoothed using Laplace correction
	 * when predictions are generated
	 */
	protected boolean m_useLaplace = false;
	
	protected RandomGenerator rng = new MersenneTwister(3071980);
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

	private boolean MinimumErrorPruning = false;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for generating a pruned or unpruned C4.5 decision tree. For more " + "information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.BOOK);
		result.setValue(Field.AUTHOR, "Ross Quinlan");
		result.setValue(Field.YEAR, "1993");
		result.setValue(Field.TITLE, "C4.5: Programs for Machine Learning");
		result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
		result.setValue(Field.ADDRESS, "San Mateo, CA");

		return result;
	}

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
	 * @param instances
	 *            the data to train the classifier with
	 * @throws Exception
	 *             if classifier can't be built successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {

		ModelSelection modSelection;

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

		if (this.getComplexityPruning()) {
//			System.out.println("complexity pruning");

			Random random = new Random(m_Seed);
			Instances cvData = new Instances(instances);
			cvData.randomize(random);
			cvData = new Instances(cvData, 0, (int) (cvData.numInstances() * m_SizePer) - 1);
			cvData.stratify(m_numFoldsPruning);

			double[][] alphas = new double[m_numFoldsPruning][];
			double[][] errors = new double[m_numFoldsPruning][];

			for (int i = 0; i < m_numFoldsPruning; i++) {

				// for every fold, grow tree on training set and fix error on
				// test set.
				Instances train = cvData.trainCV(m_numFoldsPruning, i);
				Instances test = cvData.testCV(m_numFoldsPruning, i);

				m_root.buildClassifier(train);

				// System.out.println(this.toString());
//				this.printTree();
				int numNodes = numInnerNodes();
				alphas[i] = new double[numNodes + 2];
				errors[i] = new double[numNodes + 2];

				// prune back and log alpha-values and errors on test set
				prune(alphas[i], errors[i], test);
			}

			m_root.buildClassifier(instances);

//			this.printTree();
			
			int numNodes = numInnerNodes();

			double[] treeAlphas = new double[numNodes + 2];

			// prune back and log alpha-values
			int iterations = prune(treeAlphas, null, null);
			double[] treeErrors = new double[numNodes + 2];

			for (int i = 0; i <= iterations; i++) {
				// compute midpoint alphas
				double alpha = Math.sqrt(treeAlphas[i] * treeAlphas[i + 1]);
				double error = 0;
				for (int k = 0; k < m_numFoldsPruning; k++) {
					int l = 0;
					while (alphas[k][l] <= alpha) {
						l++;
					}
					error += errors[k][l - 1];
				}
				treeErrors[i] = error / m_numFoldsPruning;
			}

			// find best alpha
			int best = -1;
			double bestError = Double.MAX_VALUE;
			for (int i = iterations; i >= 0; i--) {
				if (treeErrors[i] < bestError) {
					bestError = treeErrors[i];
					best = i;
				}
			}

			double bestAlpha = Math.sqrt(treeAlphas[best] * treeAlphas[best + 1]);

			unprune();
			prune(bestAlpha);
//			this.printTree();
		}else if(this.getMinimumErrorPruning()){
			m_root.buildClassifier(instances);
//			this.printTree();
			expectedErrorForAllLeaves();
//			this.printTree();
			minimumErrorPruning();
//			this.printTree();
		}else{
			m_root.buildClassifier(instances);
//			this.printTree();
		}

		if (m_binarySplits) {
			((BinC45ModelSelection) modSelection).cleanup();
		} else {
			((C45ModelSelection) modSelection).cleanup();
		}
	}

	private boolean getComplexityPruning() {
		return this.COMPLEXITYPRUNING;
	}

	private boolean getMinimumErrorPruning() {
		return this.MinimumErrorPruning;
	}

	private void minimumErrorPruning() {
		// TODO Auto-generated method stub
		this.m_root.minimumErrorPruning();
	}

	private void expectedErrorForAllLeaves() {
		// TODO Auto-generated method stub
		this.m_root.expectedErrorForAllLeaves();
	}

	protected void unprune() {
		this.m_root.unprune();
	}

	/**
	 * Method for performing one fold in the cross-validation of minimal
	 * cost-complexity pruning. Generates a sequence of alpha-values with error
	 * estimates for the corresponding (partially pruned) trees, given the test
	 * set of that fold.
	 * 
	 * @param alphas
	 *            array to hold the generated alpha-values
	 * @param errors
	 *            array to hold the corresponding error estimates
	 * @param test
	 *            test set of that fold (to obtain error estimates)
	 * @return the iteration of the pruning
	 * @throws Exception
	 *             if something goes wrong
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
	 * Compute sorted indices, weights and class probabilities for a given
	 * dataset. Return total weights of the data at the node.
	 * 
	 * @param data
	 *            training data
	 * @param sortedIndices
	 *            sorted indices of instances at the node
	 * @param weights
	 *            weights of instances at the node
	 * @param classProbs
	 *            class probabilities at the node
	 * @return total weights of instances at the node
	 * @throws Exception
	 *             if something goes wrong
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
	 * @param instance
	 *            the instance to classify
	 * @return the classification for the instance
	 * @throws Exception
	 *             if instance can't be classified successfully
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {

		return m_root.classifyInstance(instance);
	}

	/**
	 * Returns class probabilities for an instance.
	 * 
	 * @param instance
	 *            the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception
	 *             if distribution can't be computed successfully
	 */
	@Override
	public final double[] distributionForInstance(Instance instance) throws Exception {

		return m_root.distributionForInstance(instance, m_useLaplace);
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
	 * @return the graph describing the tree
	 * @throws Exception
	 *             if graph can't be computed
	 */
	@Override
	public String graph() throws Exception {

		return m_root.graph();
	}

	/**
	 * Returns tree in prefix order.
	 * 
	 * @return the tree in prefix order
	 * @throws Exception
	 *             if something goes wrong
	 */
	@Override
	public String prefix() throws Exception {

		return m_root.prefix();
	}

	/**
	 * Returns tree as an if-then statement.
	 * 
	 * @param className
	 *            the name of the Java class
	 * @return the tree as a Java if-then type statement
	 * @throws Exception
	 *             if something goes wrong
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
		newVector
				.addElement(
						new Option(
								"\tSet number of folds for reduced error\n"
										+ "\tpruning. One fold is used as pruning set.\n" + "\t(default 3)",
								"N", 1, "-N <number of folds>"));
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
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
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

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String seedTipText() {
		return "The seed used for randomizing the data " + "when reduced-error pruning is used.";
	}

	/**
	 * Get the value of Seed.
	 * 
	 * @return Value of Seed.
	 */
	public int getSeed() {

		return m_Seed;
	}

	/**
	 * Set the value of Seed.
	 * 
	 * @param newSeed
	 *            Value to assign to Seed.
	 */
	public void setSeed(int newSeed) {

		m_Seed = newSeed;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String useLaplaceTipText() {
		return "Whether counts at leaves are smoothed based on Laplace.";
	}

	/**
	 * Get the value of useLaplace.
	 * 
	 * @return Value of useLaplace.
	 */
	public boolean getUseLaplace() {

		return m_useLaplace;
	}

	/**
	 * Set the value of useLaplace.
	 * 
	 * @param newuseLaplace
	 *            Value to assign to useLaplace.
	 */
	public void setUseLaplace(boolean newuseLaplace) {

		m_useLaplace = newuseLaplace;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String useMDLcorrectionTipText() {
		return "Whether MDL correction is used when finding splits on numeric attributes.";
	}

	/**
	 * Get the value of useMDLcorrection.
	 * 
	 * @return Value of useMDLcorrection.
	 */
	public boolean getUseMDLcorrection() {

		return m_useMDLcorrection;
	}

	/**
	 * Set the value of useMDLcorrection.
	 * 
	 * @param newuseMDLcorrection
	 *            Value to assign to useMDLcorrection.
	 */
	public void setUseMDLcorrection(boolean newuseMDLcorrection) {

		m_useMDLcorrection = newuseMDLcorrection;
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier
	 */
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

	/**
	 * Returns a superconcise version of the model
	 * 
	 * @return a summary of the model
	 */
	@Override
	public String toSummaryString() {

		return "Number of leaves: " + m_root.numLeaves() + "\n" + "Size of the tree: " + m_root.numNodes() + "\n";
	}

	/**
	 * Returns the size of the tree
	 * 
	 * @return the size of the tree
	 */
	public double measureTreeSize() {
		return m_root.numNodes();
	}

	/**
	 * Returns the number of leaves
	 * 
	 * @return the number of leaves
	 */
	public double measureNumLeaves() {
		return m_root.numLeaves();
	}

	/**
	 * Returns the number of rules (same as number of leaves)
	 * 
	 * @return the number of rules
	 */
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
	 * @param additionalMeasureName
	 *            the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException
	 *             if the named measure is not supported
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

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String unprunedTipText() {
		return "Whether pruning is performed.";
	}

	/**
	 * Get the value of unpruned.
	 * 
	 * @return Value of unpruned.
	 */
	public boolean getUnpruned() {

		return m_unpruned;
	}

	/**
	 * Set the value of unpruned. Turns reduced-error pruning off if set.
	 * 
	 * @param v
	 *            Value to assign to unpruned.
	 */
	public void setUnpruned(boolean v) {

		if (v) {
			m_reducedErrorPruning = false;
		}
		m_unpruned = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String collapseTreeTipText() {
		return "Whether parts are removed that do not reduce training error.";
	}

	/**
	 * Get the value of collapseTree.
	 * 
	 * @return Value of collapseTree.
	 */
	public boolean getCollapseTree() {

		return m_collapseTree;
	}

	/**
	 * Set the value of collapseTree.
	 * 
	 * @param v
	 *            Value to assign to collapseTree.
	 */
	public void setCollapseTree(boolean v) {

		m_collapseTree = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String confidenceFactorTipText() {
		return "The confidence factor used for pruning (smaller values incur " + "more pruning).";
	}

	/**
	 * Get the value of CF.
	 * 
	 * @return Value of CF.
	 */
	public float getConfidenceFactor() {

		return m_CF;
	}

	/**
	 * Set the value of CF.
	 * 
	 * @param v
	 *            Value to assign to CF.
	 */
	public void setConfidenceFactor(float v) {

		m_CF = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String minNumObjTipText() {
		return "The minimum number of instances per leaf.";
	}

	/**
	 * Get the value of minNumObj.
	 * 
	 * @return Value of minNumObj.
	 */
	public int getMinNumObj() {

		return m_minNumObj;
	}

	/**
	 * Set the value of minNumObj.
	 * 
	 * @param v
	 *            Value to assign to minNumObj.
	 */
	public void setMinNumObj(int v) {

		m_minNumObj = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String reducedErrorPruningTipText() {
		return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
	}

	/**
	 * Get the value of reducedErrorPruning.
	 * 
	 * @return Value of reducedErrorPruning.
	 */
	public boolean getReducedErrorPruning() {

		return m_reducedErrorPruning;
	}

	/**
	 * Set the value of reducedErrorPruning. Turns unpruned trees off if set.
	 * 
	 * @param v
	 *            Value to assign to reducedErrorPruning.
	 */
	public void setReducedErrorPruning(boolean v) {

		if (v) {
			m_unpruned = false;
		}
		m_reducedErrorPruning = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numFoldsTipText() {
		return "Determines the amount of data used for reduced-error pruning. "
				+ " One fold is used for pruning, the rest for growing the tree.";
	}

	/**
	 * Get the value of numFolds.
	 * 
	 * @return Value of numFolds.
	 */
	public int getNumFolds() {

		return m_numFolds;
	}

	/**
	 * Set the value of numFolds.
	 * 
	 * @param v
	 *            Value to assign to numFolds.
	 */
	public void setNumFolds(int v) {

		m_numFolds = v;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String binarySplitsTipText() {
		return "Whether to use binary splits on nominal attributes when " + "building the trees.";
	}

	/**
	 * Get the value of binarySplits.
	 * 
	 * @return Value of binarySplits.
	 */
	public boolean getBinarySplits() {

		return m_binarySplits;
	}

	/**
	 * Set the value of binarySplits.
	 * 
	 * @param v
	 *            Value to assign to binarySplits.
	 */
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
	 * @param v
	 *            Value to assign to subtreeRaising.
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
	 * @param v
	 *            true if instance data is to be saved
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
	 * @param m_doNotMakeSplitPointActualValue
	 *            the value to set
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

	/**
	 * Returns the number of elements in the partition.
	 */
	@Override
	public int numElements() throws Exception {

		return m_root.numNodes();
	}

	private void printTree() {
		System.out.println(m_root.printNks("root"));
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
	
	private void setMinimumErrorPruning(boolean v) {
		if(v){
			this.setReducedErrorPruning(false);
			this.setComplexityPruning(false);
		}
		this.MinimumErrorPruning  = v;
	}

	private void setComplexityPruning(boolean v) {
		if(v){
			this.setReducedErrorPruning(false);
			this.setMinimumErrorPruning(false);
		}
		this.COMPLEXITYPRUNING = v;
	}

	/**
	 * Main method for testing this class
	 * 
	 * @param argv
	 *            the commandline options
	 * @throws Exception
	 */
	public static void main(String[] argv) throws Exception {

		System.out.println(Arrays.toString(argv));

		String tmpStr = Utils.getOption('t', argv);
		File folder = new File(tmpStr);
		File[] listOfFiles = folder.listFiles();

		for (int n = 0; n < 66; n++) {
			if (listOfFiles[n].isFile()) {
				File file = listOfFiles[n];
				FileReader fr = new FileReader(file);
				Instances data = new Instances(fr);
				data.setClassIndex(data.numAttributes() - 1);

				System.out.print(file.getName() + "\t");

//				C45CostComplexityPruning tree1 = new C45CostComplexityPruning();
//				tree1.m_unpruned = true;
//				tree1.m_useLaplace = false;
//				tree1.COMPLEXITYPRUNING = false;
//				tree1.m_numFoldsPruning = 2;

				// double start = System.currentTimeMillis();
//				tree1.buildClassifier(data);
				// double trainingTime= System.currentTimeMillis()-start;
				// System.out.println("time: "+trainingTime);
				// tree1.printTree();
				// System.out.println(tree1.toString());
				// System.out.println(data.firstInstance().toString());
				// double[] res =
				// tree1.distributionForInstance(data.firstInstance());
				// System.out.println(Arrays.toString(res));

//				 long seed = 3071980;
				 long seed = 2511990;
				 double rmse = 0, error = 0;
				 for (int exp = 0; exp < 5; exp++) {
				// System.out.print("*");
				 Random rg = new Random(seed);
				
				C45CostComplexityPruning tree = new C45CostComplexityPruning();

				tree.setReducedErrorPruning(true);
//				 tree.setUseLaplace(true);
//				 tree.setComplexityPruning(true);
//				 tree.setMinimumErrorPruning(true);

				 Evaluation eva = new Evaluation(data);
				 eva.crossValidateModel(tree, data, 2, rg);
				
				 rmse += eva.rootMeanSquaredError();
				 error += eva.errorRate();
				 }
				 rmse /= 5;
				 error /= 5;
				 System.out.println("\t" + Utils.doubleToString(rmse, 4) +
				 "\t" + Utils.doubleToString(error, 4));
			}
		}
	}
}

