package tree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import method.HDPMethod;
import method.SmoothingMethod;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public class TenFoldCVTreeSmoothing {

	private static SmoothingMethod method;
	private static String data;
	private static boolean m_unPruning;
	private static String m_Tying;
	private static int m_GibbsIteration;

	private static HDPMethod methodHDP;

	public static void main(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		setOptions(args);

		if (data.isEmpty()) {
			System.err.println("No Training File given");
			System.exit(-1);
		}

		File sourceFile = new File(data);
		if (!sourceFile.exists()) {
			System.err.println("File " + data + " not found!");
			System.exit(-1);
		}

//		File[] folder = sourceFile.listFiles();
//		Arrays.sort(folder);
//		for (int d = 109; d < folder.length; d++) {

//			sourceFile = folder[d];
			String name = sourceFile.getName().substring(0, sourceFile.getName().indexOf("."));
			System.out.print(name);
			BufferedReader reader = new BufferedReader(new FileReader(sourceFile));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			int nD = data.numInstances();
			int nA = data.numAttributes();
			int nC = data.numClasses();
			System.out.print("\t"+nD+"\t"+(nA-1)+"\t"+nC);
			long seed = 25011990;

			Random random = new Random(seed);
			C45 tree = new C45();
//			 tree.setReducedErrorPruning(true);
			// tree.setComplexityPruning(true);
			tree.setUnpruned(m_unPruning);
			tree.setMethod(method);

			if(method == SmoothingMethod.HDP){
				tree.setTyingStrategy(m_Tying);
				tree.setGibbsIteration(m_GibbsIteration);
				tree.setHDPMethod(methodHDP);
			}
//			tree.buildClassifier(data);
//			System.out.println(tree.toString());
			
			EvaluationC45 eva = new EvaluationC45(data);
			eva.crossValidateModel(tree, data, 10, random);
			
//			System.out.print("\t"+nD+"\t"+(nA-1)+"\t"+nC);
			System.out.print("\t"+Utils.doubleToString(eva.rootMeanSquaredError(), 6, 4));
			System.out.print("\t"+Utils.doubleToString(eva.errorRate(), 6,4));	
			System.out.print("\t"+eva.getTrainTime()+"\n");
			System.out.println();
//		}
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
			}else if(string.equalsIgnoreCase("RECURSIVE")) {
				method = SmoothingMethod.RECURSIVE;
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
		
		Utils.checkForRemainingOptions(options);
	}

}
