package weka.classifiers.mmall.Ensemble;

import java.io.File;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class PhonemBackward {
	
	public static int STEP = 0;

	public static void main(String[] args) throws Exception {
		DataSource source = new DataSource("/home/petitjean/Dropbox/Data/datasets_discretized_and_MissingHandled/phoneme.arff");
		Instances originalData = source.getDataSet();
		if (originalData.classIndex() == -1) {
			originalData.setClassIndex(originalData.numAttributes() - 1);
		}
		Instances data = new Instances(originalData);
		// delete missing classValues
		int nClassValues = originalData.numClasses();

		int[] classCounts = new int[nClassValues];
		for (int i = 0; i < originalData.numInstances(); i++) {
			Instance inst = originalData.get(i);
			classCounts[(int) inst.classValue()]++;
		}
		System.out.println(Arrays.toString(classCounts));

		String[] optionsLR = new String[] { "-D", "-S", "A2JE", "-P", "dCCBN", "-I", "Flat" };
		String[] optionsWanbia = new String[] { "-D", "-S", "A2JE", "-P", "wCCBN", "-I", "Flat" };
		double errorLR = getErrorRate(data, true);
		double wanbiaError = getErrorRate(data, false);
		System.out.println("LR=" + errorLR + "\twanbia=" + wanbiaError);
		while (Math.abs(wanbiaError - errorLR) > .05) {
			System.out.println("LR=" + errorLR + "\twanbia=" + wanbiaError);
			// find second least frequent class value
			double smallestFreq = 100;
			int indexClassLowestFreq = -1;
			for (int c = 0; c < classCounts.length; c++) {
				if (classCounts[c] > 2 && classCounts[c] < smallestFreq) {
					smallestFreq = classCounts[c];
					indexClassLowestFreq = c;
				}
			}

			System.out.println("deleting all instances with class value #" + indexClassLowestFreq);
			Instances copyData = new Instances(data);
			int nDeletions = 0;
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.get(i);
				if (((int) inst.classValue()) == indexClassLowestFreq) {
					copyData.delete(i - nDeletions);
					nDeletions++;
				}
			}
			ArffSaver saver = new ArffSaver();
			saver.setInstances(copyData);
			saver.setFile(new File("/home/petitjean/phoneme-" + STEP + ".arff"));
			saver.writeBatch();
			source = new DataSource("/home/petitjean/phoneme-" + STEP + ".arff");
			data = source.getDataSet();
			if (data.classIndex() == -1) {
				data.setClassIndex(originalData.numAttributes() - 1);
			}
			Arrays.fill(classCounts, 0);
			for (int i = 0; i < data.numInstances(); i++) {
				Instance inst = data.get(i);
				classCounts[(int) inst.classValue()]++;
			}
			System.out.println(Arrays.toString(classCounts));
			errorLR = getErrorRate(data, true);
			wanbiaError = getErrorRate(data, false);
			STEP++;

		}
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File("/home/petitjean/phoneme-simplified.arff"));
		saver.writeBatch();

	}

	public static double getErrorRate(Instances data, boolean LR) throws Exception {
		Instances randData = new Instances(data);
		randData.randomize(new Random(STEP));
		double errorRate = 0.0;
		int nEvals = 0;
		for (int n = 0; n < 2; n++) {
			Instances train = randData.trainCV(2, n);
			Instances test = randData.testCV(2, n);
			Evaluation eval = new Evaluation(randData);
			Classifier classifier;
			if (LR) {
				classifier = AbstractClassifier.forName("weka.classifiers.mmall.Ensemble.wdAnJE",
								new String[] { "-D", "-S", "A2JE", "-P", "dCCBN", "-I", "Flat" });
			}else{
				classifier = AbstractClassifier.forName("weka.classifiers.mmall.Ensemble.wdAnJE",
								new String[] { "-D", "-S", "A2JE", "-P", "wCCBN", "-I", "Flat" });
			}
			classifier.buildClassifier(train);
			eval.evaluateModel(classifier, test, new String[0]);
			errorRate += eval.errorRate();
			nEvals++;
		}
		errorRate /= nEvals;
		return errorRate;
	}

}
