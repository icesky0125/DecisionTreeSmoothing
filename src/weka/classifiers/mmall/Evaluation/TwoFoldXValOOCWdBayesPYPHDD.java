package weka.classifiers.mmall.Evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;
import org.apache.commons.math3.random.MersenneTwister;

import method.Method;
import pyp.ProbabilityTree;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.mmall.Online.Bayes.wdBayesOnlinePYP;
import weka.classifiers.mmall.Online.Bayes.wdBayesOnlinePYP_Penny;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.Saver;

public class TwoFoldXValOOCWdBayesPYPHDD {

	private static String data = "";
	private static String m_OutputResults = "";
	private static String m_S = "KDB"; // -S (NB, TAN,
										// KDB,UpperKDB,LowerKDB,AllOrderKDB)
	private static int m_K = 5; // -K
	private static boolean m_MVerb = false; // -V
	private static Instances instances = null;
	private static int m_nExp = 5;
	public static final int BUFFER_SIZE = 10 * 1024 * 1024; // 100MB
	private static int m_IterGibbs;
	private static boolean M_estimation = false; // default: using HDP
	private static double m_Discount = 0;
	private static int m_EnsembleSize = 5;
	private static boolean m_Backoff = true;
	private static int m_Tying = 1;
	private static boolean LOOCV = false;
	protected static Method method = Method.LOOCV;

	public static void main(String[] args) throws Exception {

		System.out.println(Arrays.toString(args));
		
		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File source;
		source = new File(data);
		if (!source.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}
		
		File file = new File(data);
//		for(int i = 0; i <1; i++){
		for(int i = 0; i <source.length() ; i++){
			File sourceFile = source.listFiles()[i];
//			File sourceFile = new File(data);
			System.out.print(sourceFile.getName() + "\t");
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
			Instances structure = reader.getStructure();
			structure.setClassIndex(structure.numAttributes() - 1);
			int nc = structure.numClasses();
			int N = getNumData(sourceFile, structure);
			if (m_MVerb) {
				System.out.println("read "+N+" datapoints");
			}
			
			double m_RMSE = 0;
			double m_Error = 0;
			int NTest = 0;
			long seed = 3071980;
			long trainTime = 0;
			/*
			 * Start m_nExp rounds of Experiments
			 */
//			for (int exp = 0; exp < 1; exp++) {
			for (int exp = 0; exp < m_nExp; exp++) {
				System.out.print("*");

				MersenneTwister rg = new MersenneTwister(seed);
				BitSet test0Indexes = getTest0Indexes(sourceFile, structure, rg);
				// ---------------------------------------------------------
				// Train on Fold 0
				// ---------------------------------------------------------				
				wdBayesOnlinePYP_Penny learner = new wdBayesOnlinePYP_Penny();
				learner.set_m_S(m_S);
				learner.setK(m_K);
				learner.setRandomGenerator(rg);
				learner.setMethod(method);
				learner.setGibbsIteration(m_IterGibbs);
				learner.setDiscount(m_Discount);
				learner.setEnsembleSize(m_EnsembleSize);
				learner.setBackoff(m_Backoff);
				learner.setM_Tying(m_Tying);
			
//				learner.buildClassifier(sourceFile);
//				Instance ins = reader.readInstance(structure);
//				double[] res = learner.distributionForInstance(ins);
//				System.out.println(Arrays.toString(res));
				// creating tempFile for train0
				File trainFile = createTrainTmpFile(sourceFile, structure, test0Indexes);
//				 System.out.println("generated train");
				if (m_MVerb) {
					System.out.println("Training fold 0: trainFile is '" + trainFile.getAbsolutePath() + "'");
				}
				
				long start = System.currentTimeMillis();
				learner.buildClassifier(trainFile);
				start = System.currentTimeMillis()-start;
				trainTime += start;
//				System.out.print(start+"\t");
				
				if (m_MVerb) {
					System.out.println("time\t"+(System.currentTimeMillis()-start));
				}
				
				// ---------------------------------------------------------
				// Test on Fold 1
				// ---------------------------------------------------------
				if (m_MVerb) {
					System.out.println("Testing fold 0 started");
				}

//				start = System.currentTimeMillis();
				int lineNo = 0;
				Instance current;
				int thisNTest = 0;
				
				reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
				long testTime = System.currentTimeMillis();
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

//				System.out.println(NTest);
				
				
				if (m_MVerb) {
					System.out.println("test time:"+(System.currentTimeMillis()-start));
					System.out.println(
							"Testing fold 0 finished - 0-1=" + Utils.doubleToString(m_Error / NTest, 6, 4) + "\trmse=" + Utils.doubleToString(Math.sqrt(m_RMSE / NTest),6,4));
			
				}

				if (Math.abs(thisNTest - test0Indexes.cardinality()) > 1) {
					System.err.println("no! " + thisNTest + "\t" + test0Indexes.cardinality());
				}

				BitSet test1Indexes = new BitSet(lineNo);
				test1Indexes.set(0, lineNo);
				test1Indexes.xor(test0Indexes);
				

				// ---------------------------------------------------------
				// Train on Fold 1
				// ---------------------------------------------------------
				learner = new wdBayesOnlinePYP_Penny();
				learner.set_m_S(m_S);
				learner.setK(m_K);
				learner.setRandomGenerator(rg);
				learner.setMethod(method);
//				learner.setMEstimation(M_estimation);
				learner.setGibbsIteration(m_IterGibbs);
				learner.setDiscount(m_Discount);
				learner.setEnsembleSize(m_EnsembleSize);
				learner.setBackoff(m_Backoff);
				learner.setM_Tying(m_Tying);
//				learner.setLOOCV(LOOCV);
//				learner.buildClassifier(sourceFile);
//				Instance ins = reader.readInstance(structure);
//				learner.distributionForInstance(ins);
				
				// creating tempFile for train0
				trainFile = createTrainTmpFile(sourceFile, structure, test1Indexes);

//				if (m_MVerb) {
//					System.out.println("Training fold 1: trainFile is '" + trainFile.getAbsolutePath() + "'");
//				}
				start = System.currentTimeMillis();
				learner.buildClassifier(trainFile);
				start = System.currentTimeMillis()-start;
				trainTime += start;
//				System.out.print(start+"\t");
				
				// ---------------------------------------------------------
				// Test on Fold 0
				// ---------------------------------------------------------
//				if (m_MVerb) {
//					System.out.println("Testing fold 0 started");
//				}

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
//				System.out.println(NTest);
				if (m_MVerb) {
					System.out.println("Testing exp " + exp + " fold 1 finished - 0-1=" + Utils.doubleToString(m_Error / NTest, 6, 4) + "\trmse="
							+ Utils.doubleToString(Math.sqrt(m_RMSE / NTest),6,4));
				}

				seed++;
			} // Ends No. of Experiments
//			
//			 System.out.print("\nBias-Variance Decomposition\n");
//			 System.out.print("\nClassifier : wdBayesOnline_" + m_S);
//			 String strData =
//			 data.substring(data.lastIndexOf("/")+1,data.lastIndexOf("."));
//			 System.out.print( "\nData File : " + data);
			trainTime = (long)trainTime/10;
			System.out.println("\t"+Utils.doubleToString(Math.sqrt(m_RMSE / NTest), 6, 4) + "\t"
					+ Utils.doubleToString(m_Error / NTest, 6, 4)+"\t"+Utils.doubleToString(trainTime, 6, 4));
			
		}
	}

	public static int ind(int i, int j) {
		return (i == j) ? 1 : 0;
	}

	public static void setOptions(String[] options) throws Exception {

		String string;
		
		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
		}
		
		string = Utils.getOption('m', options);
		if (string.length() != 0) {
			int num = Integer.parseInt(string);
			if (num == 1){
				method = Method.LOOCV;
			}else if( num == 2){
				method = Method.HDP;
			}else{
				method = Method.M_estimation;	
			}
		}
		
		string = Utils.getOption('I', options);
		if (string.length() != 0) {
			m_IterGibbs = Integer.parseInt(string);
		}

		string = Utils.getOption('D', options);
		if (string.length() != 0) {
			m_Discount = Double.parseDouble(string);
		}
		
		String ML = Utils.getOption('L', options);
		if (ML.length() != 0) {
			m_Tying = Integer.parseInt(ML);
		}
		
		m_Backoff = Utils.getFlag('B', options);
		
		string = Utils.getOption('O', options);
		if (string.length() != 0) {
			m_OutputResults = string;
		}

		m_MVerb = Utils.getFlag('V', options);
//		M_estimation = Utils.getFlag('M', options);
//		LOOCV = Utils.getFlag('C', options);
//		m_Backoff  = Utils.getFlag('B', options);
		
		string = Utils.getOption('S', options);
		if (string.length() != 0) {
			m_S = string;
		}

		string = Utils.getOption('K', options);
		if (string.length() != 0) {
			m_K = Integer.parseInt(string);
		}
		
		string = Utils.getOption('E', options);
		if (string.length() != 0) {
			m_EnsembleSize = Integer.parseInt(string);
		}

		string = Utils.getOption('X', options);
		if (string.length() != 0) {
			m_nExp = Integer.valueOf(string);
		}

		
		Utils.checkForRemainingOptions(options);
	}

	private static int getNumData(File sourceFile, Instances structure) throws FileNotFoundException, IOException {
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
		int nLines = 0;
		while (reader.readInstance(structure) != null) {
//			if (nLines % 1000000 == 0) {
//				System.out.println(nLines);
//			}
			nLines++;
		}
		return nLines;
	}

	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg) throws FileNotFoundException, IOException {
		BitSet res = new BitSet();
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);
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

	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
		File out = File.createTempFile("train-", ".arff");
		out.deleteOnExit();
		ArffSaver fileSaver = new ArffSaver();
		fileSaver.setFile(out);
		fileSaver.setRetrieval(Saver.INCREMENTAL);
		fileSaver.setStructure(structure);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile),BUFFER_SIZE), 100000);

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

	public static String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}
	
//	private static BitSet getTest0Indexes(File sourceFile, Instances structure, MersenneTwister rg)
//			throws FileNotFoundException, IOException {
//		BitSet res = new BitSet();
//		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
//		int nLines = 0;
//		while (reader.readInstance(structure) != null) {
//			if (rg.nextBoolean()) {
//				res.set(nLines);
//			}
//			nLines++;
//		}
//
//		int expectedNLines = (nLines % 5 == 0) ? nLines / 5 : nLines / 5 + 1;
//		int actualNLines = res.cardinality();
//		if (actualNLines < expectedNLines) {
//			while (actualNLines < expectedNLines) {
//				int chosen;
//				do {
//					chosen = rg.nextInt(nLines);
//				} while (res.get(chosen));
//				res.set(chosen);
//				actualNLines++;
//			}
//		} else if (actualNLines > expectedNLines) {
//			while (actualNLines > expectedNLines) {
//				int chosen;
//				do {
//					chosen = rg.nextInt(nLines);
//				} while (!res.get(chosen));
//				res.clear(chosen);
//				actualNLines--;
//			}
//		}
//		return res;
//	}
//
//	public static File createTrainTmpFile(File sourceFile, Instances structure, BitSet testIndexes) throws IOException {
//		File out = File.createTempFile("train-", ".arff");
//		out.deleteOnExit();
//		ArffSaver fileSaver = new ArffSaver();
//		fileSaver.setFile(out);
//		fileSaver.setRetrieval(Saver.INCREMENTAL);
//		fileSaver.setStructure(structure);
//
//		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile), BUFFER_SIZE), 100000);
//
//		Instance current;
//		int lineNo = 0;
//		while ((current = reader.readInstance(structure)) != null) {
//			if (!testIndexes.get(lineNo)) {
//				fileSaver.writeIncremental(current);
//			}
//			lineNo++;
//		}
//		fileSaver.writeIncremental(null);
//		return out;
//	}
//
//	public static String[] getOptions() {
//		String[] options = new String[3];
//		int current = 0;
//		while (current < options.length) {
//			options[current++] = "";
//		}
//		return options;
//	}

}
