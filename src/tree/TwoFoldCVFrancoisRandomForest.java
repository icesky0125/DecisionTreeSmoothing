package tree;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;

import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;
import java.util.BitSet;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader;

public class TwoFoldCVFrancoisRandomForest {

	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static String data;

	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));

		setOptions(args);

		File folder = new File(data);
		File[] listOfFiles = folder.listFiles();
		Arrays.sort(listOfFiles);

		for (int i = 0; i < listOfFiles.length; i++) {
			File sourceFile = listOfFiles[i];
//		File sourceFile = new File(data);
			System.out.print(sourceFile.getName() + "\t");
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);
			int nc = structure.numClasses();

			double m_RMSE = 0;
			double m_Error = 0;
			int NTest = 0;
			long seed = 3071980;
			long trainTime = 0;
			/*
			 * Start m_nExp rounds of Experiments
			 */
			for (int exp = 0; exp < 5; exp++) {
				// System.out.print("*");

				MersenneTwister rg = new MersenneTwister(seed);
				BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);
				// ---------------------------------------------------------
				// Train on Fold 0
				// ---------------------------------------------------------
				RandomForest learner = new RandomForest();
				learner.setNumTrees(100);

				// creating tempFile for train0
				File trainFile = createTrainTmpFile(sourceFile, structure, test0Indexes);
				// System.out.println("generated train");
				BufferedReader reader1 = new BufferedReader(new FileReader(trainFile));
				// BufferedReader reader1 = new BufferedReader(new
				// FileReader(sourceFile));
				ArffReader arff = new ArffReader(reader1);
				Instances train = arff.getData();
				train.setClassIndex(train.numAttributes() - 1);
				// System.out.println(train.numInstances());
				long start = System.currentTimeMillis();
				learner.buildClassifier(train);
				start = System.currentTimeMillis() - start;
				trainTime += start;
				// System.out.print(start+"\t");
				// System.out.println();
				// System.out.println(learner.toString());

				// ---------------------------------------------------------
				// Test on Fold 1
				// ---------------------------------------------------------

				start = System.currentTimeMillis();
				int lineNo = 0;
				Instance current;
				int thisNTest = 0;

				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);

				while ((current = reader.readInstance(structure)) != null) {
					if (test0Indexes.get(lineNo)) {
						double[] probs = new double[nc];
						probs = learner.distributionForInstance(current);
						int x_C = (int) current.classValue();

						// ------------------------------------
						// Update Error and RMSE
						// ------------------------------------
						int pred = -1;
						double bestProb = Double.MIN_VALUE;
						for (int y = 0; y < nc; y++) {
							if (!Double.isNaN(probs[y])) {
								if (probs[y] > bestProb) {
									pred = y;
									bestProb = probs[y];
								}
								m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
							} else {
								System.err.println("probs[ " + y + "] is NaN! oh no!");
							}
						}

						if (pred != x_C) {
							m_Error += 1;
						}

						thisNTest++;
						NTest++;
					}
					lineNo++;
				}

				// System.out.println(NTest);

				if (Math.abs(thisNTest - test0Indexes.cardinality()) > 1) {
					System.err.println("no! " + thisNTest + "\t" + test0Indexes.cardinality());
				}

				BitSet test1Indexes = new BitSet(lineNo);
				test1Indexes.set(0, lineNo);
				test1Indexes.xor(test0Indexes);

				// ---------------------------------------------------------
				// Train on Fold 1
				// ---------------------------------------------------------
				learner = new RandomForest();
				learner.setNumTrees(100);

				// creating tempFile for train0
				trainFile = createTrainTmpFile(sourceFile, structure, test1Indexes);

				// if (m_MVerb) {
				// System.out.println("Training fold 1: trainFile is '" +
				// trainFile.getAbsolutePath() + "'");
				// }
				reader1 = new BufferedReader(new FileReader(trainFile));
				arff = new ArffReader(reader1);
				train = arff.getData();
				train.setClassIndex(train.numAttributes() - 1);

				start = System.currentTimeMillis();
				learner.buildClassifier(train);
				start = System.currentTimeMillis() - start;
				trainTime += start;
				// ---------------------------------------------------------
				// Test on Fold 0
				// ---------------------------------------------------------
				// if (m_MVerb) {
				// System.out.println("Testing fold 0 started");
				// }

				lineNo = 0;
				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
				while ((current = reader.readInstance(structure)) != null) {
					if (test1Indexes.get(lineNo)) {
						double[] probs = new double[nc];
						probs = learner.distributionForInstance(current);
						int x_C = (int) current.classValue();

						// ------------------------------------
						// Update Error and RMSE
						// ------------------------------------
						int pred = -1;
						double bestProb = Double.MIN_VALUE;
						for (int y = 0; y < nc; y++) {
							if (!Double.isNaN(probs[y])) {
								if (probs[y] > bestProb) {
									pred = y;
									bestProb = probs[y];
								}
								m_RMSE += (1 / (double) nc * Math.pow((probs[y] - ((y == x_C) ? 1 : 0)), 2));
							} else {
								System.err.println("probs[ " + y + "] is NaN! oh no!");
							}
						}

						if (pred != x_C) {
							m_Error += 1;
						}
						NTest++;
					}
					lineNo++;
				}
				// System.out.println(NTest);

				seed++;
			} // Ends No. of Experiments
				//
				// System.out.print("\nBias-Variance Decomposition\n");
				// System.out.print("\nClassifier : wdBayesOnline_" + m_S);
				// String strData =
				// data.substring(data.lastIndexOf("/")+1,data.lastIndexOf("."));
				// System.out.print( "\nData File : " + data);
			trainTime = (long) trainTime / 10;
			System.out.println(Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4) + "\t"
					+ Utils.doubleToString(m_Error / NTest, 6, 4) + "\t" + Utils.doubleToString(trainTime, 6, 4));
		}

	}

	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
		File out = File.createTempFile("train-", ".arff");
		out.deleteOnExit();
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);

		Instance current;
		int lineNo = 0;
		while ((current = reader.readInstance(structure)) != null) {
			if (!testIndexes.get(lineNo)) {
				fileSaver.writeIncremental(current);
			}
			lineNo++;
		}
		fileSaver.writeIncremental(null);
		return out;
	}

	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg)
			throws FileNotFoundException, IOException {
		BitSet res = new BitSet();
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
			if (rg.nextBoolean()) {
				res.set(nLines);
			}
			nLines++;
		}

		int expectedNLines = (nLines % 2 == 0) ? nLines / 2 : nLines / 2 + 1;
		int actualNLines = res.cardinality();

		if (actualNLines < expectedNLines) {
			while (actualNLines < expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (res.get(chosen));
				res.set(chosen);
				actualNLines++;
			}
		} else if (actualNLines > expectedNLines) {
			while (actualNLines > expectedNLines) {
				int chosen;
				do {
					chosen = rg.nextInt(nLines);
				} while (!res.get(chosen));
				res.clear(chosen);
				actualNLines--;
			}
		}
		return res;
	}

	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
		}

		Utils.checkForRemainingOptions(options);
	}

}
