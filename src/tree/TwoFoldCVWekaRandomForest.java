package tree;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import method.HDPMethod;
import method.SmoothingMethod;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;

public class TwoFoldCVWekaRandomForest {

	private static SmoothingMethod method;
	private static String data;
	private static boolean m_unPruning;
	private static Integer m_nExp;
	private static String m_Tying;
	private static int m_GibbsIteration;
	
	private static HDPMethod methodHDP;
	private static int nExp =5;

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println(Arrays.toString(args));

		setOptions(args);
		 File folder = new File(data);
		 File[] listOfFiles = folder.listFiles();
		 Arrays.sort(listOfFiles);
		 
		for (int n = 0; n < listOfFiles.length; n++) {
		if (listOfFiles[n].isFile()) {

			File file = listOfFiles[n];
//			System.out.println(file.getName());
		
//			 File file = new File(data);
			FileReader fr = new FileReader(file);
			Instances data = new Instances(fr);
			data.setClassIndex(data.numAttributes() - 1);
			System.out.print(file.getName() + "\t"+data.numInstances()+"\t"+data.numAttributes()+"\t"+data.numClasses());
//			System.out.println();
			long seed = 3071980;
			double rmse = 0, error = 0;
			
			for (int exp = 0; exp < nExp; exp++) {

				Random rg = new Random(seed);
				RandomForest rf = new RandomForest();
				rf.setNumTrees(8);
//				C45 tree = new C45();
//				 tree.setReducedErrorPruning(true);
				// tree.setComplexityPruning(true);
//				tree.setUnpruned(m_unPruning);
//				tree.setMethod(method);
//
//				if(method == Method.HDP){
//					tree.setTyingStrategy(m_Tying);
//					tree.setGibbsIteration(m_GibbsIteration);
//					tree.setHDPMethod(methodHDP);
//				}
				
				Evaluation eva = new Evaluation(data);
				eva.crossValidateModel(rf, data, 2, rg);

				rmse += eva.rootMeanSquaredError();
				error += eva.errorRate();
			}
			rmse /= 5;
			error /= 5;
			System.out.println(Utils.doubleToString(rmse, 4) + "\t" + Utils.doubleToString(error, 4));
		}
	}
	}
	
	public static void setOptions(String[] options) throws Exception {

		String string;

		string = Utils.getOption('t', options);
		if (string.length() != 0) {
			data = string;
		}
		
		string = Utils.getOption('M', options);
		if (string.length() != 0) {
			
			if(string.equalsIgnoreCase("None")) {
				method = SmoothingMethod.None;
			}else if(string.equalsIgnoreCase("LAPLACE")) {
				method = SmoothingMethod.LAPLACE;
			}else if(string.equalsIgnoreCase("M_estimation")) {
				method = SmoothingMethod.M_estimation;
			}else if(string.equalsIgnoreCase("HDP")) {
				method = SmoothingMethod.HDP;
			}else if(string.equalsIgnoreCase("HGS")) {
				method = SmoothingMethod.HGS;
			}else if(string.equalsIgnoreCase("OptiMestimation")) {
				method = SmoothingMethod.OptiMestimation;
			}else if(string.equalsIgnoreCase("VALIDATE")) {
				method = SmoothingMethod.VALIDATE;
			}else {
				System.out.println("no this method found!");
			}
		}
		
		string = Utils.getOption('H', options);
		if (string.length() != 0) {
			
			if(string.equalsIgnoreCase("Expected")) {
				methodHDP = HDPMethod.Expected;
			}else if(string.equalsIgnoreCase("alpha")){
				methodHDP = HDPMethod.Alpha;
			}else {
				System.out.println("no such method found!");
			}
		}
		
		m_unPruning = Utils.getFlag('P', options);


		string = Utils.getOption('T', options);
		if (string.length() != 0) {
			m_Tying = string;
		}
		
		string = Utils.getOption('G', options);
		if (string.length() != 0) {
			m_GibbsIteration = Integer.parseInt(string);
		}
		
		string = Utils.getOption('X', options);
		if (string.length() != 0) {
			nExp = Integer.parseInt(string);
		}
		
		Utils.checkForRemainingOptions(options);
	}
}
