package tree;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;

public class EvaluationC45 extends Evaluation {
	
	double trainTime = 0;

	public EvaluationC45(Instances data) throws Exception {
		super(data);
		// TODO Auto-generated constructor stub
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	 /**
	   * Performs a (stratified if class is nominal) cross-validation for a
	   * classifier on a set of instances. Now performs a deep copy of the
	   * classifier before each call to buildClassifier() (just in case the
	   * classifier is not initialized properly).
	   * 
	   * @param classifier the classifier with any options set.
	   * @param data the data on which the cross-validation is to be performed
	   * @param numFolds the number of folds for the cross-validation
	   * @param random random number generator for randomization
	   * @param forPredictionsPrinting varargs parameter that, if supplied, is
	   *          expected to hold a
	   *          weka.classifiers.evaluation.output.prediction.AbstractOutput
	   *          object
	   * @throws Exception if a classifier could not be generated successfully or
	   *           the class is not defined
	   */
	  public void crossValidateModel(Classifier classifier, Instances data,
	    int numFolds, Random random, Object... forPredictionsPrinting)
	    throws Exception {

	    // Make a copy of the data we can reorder
	    data = new Instances(data);
	    data.randomize(random);
	    if (data.classAttribute().isNominal()) {
	      data.stratify(numFolds);
	    }

	    // We assume that the first element is a
	    // weka.classifiers.evaluation.output.prediction.AbstractOutput object
	    AbstractOutput classificationOutput = null;
	    if (forPredictionsPrinting.length > 0) {
	      // print the header first
	      classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
	      classificationOutput.setHeader(data);
	      classificationOutput.printHeader();
	    }

	    // Do the folds
	    for (int i = 0; i < numFolds; i++) {
	      Instances train = data.trainCV(numFolds, i, random);
	      setPriors(train);
	      Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
	      double start = System.currentTimeMillis();
	      copiedClassifier.buildClassifier(train);
	      trainTime += System.currentTimeMillis()-start;
	      Instances test = data.testCV(numFolds, i);
	      evaluateModel(copiedClassifier, test, forPredictionsPrinting);
	    }
	    m_NumFolds = numFolds;
	    trainTime /= numFolds;
	    if (classificationOutput != null) {
	      classificationOutput.printFooter();
	    }
	  }

	  public double getTrainTime() {
		  return this.trainTime;
	  }

}
