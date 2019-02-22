package tree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

import method.GradientMethod;
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
	private static GradientMethod methodG;

	public static void main(String[] args) throws Exception {
//		System.setOut(new PrintStream(new FileOutputStream("output.txt")));
//		System.out.println("This is test output");
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

		File[] folder = sourceFile.listFiles();
		Arrays.sort(folder);
		for (int d = 70; d < folder.length; d++) {

			sourceFile = folder[d];
			String name = sourceFile.getName().substring(0, sourceFile.getName().indexOf("."));
			System.out.print(name);
			BufferedReader reader = new BufferedReader(new FileReader(sourceFile));
			ArffReader arff = new ArffReader(reader);
			Instances data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			int nD = data.numInstances();
			int nA = data.numAttributes();
			int nC = data.numClasses();
			
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
			
			if(method == SmoothingMethod.HGS) {
				tree.setGradientMethod(methodG);
			}
			
			EvaluationC45 eva = new EvaluationC45(data);
			eva.crossValidateModel(tree, data, 10, random);
			
//			System.out.print("\t"+nD+"\t"+(nA-1)+"\t"+nC);
			System.out.print("\t"+Utils.doubleToString(eva.rootMeanSquaredError(), 6, 4));
			System.out.print("\t"+Utils.doubleToString(eva.errorRate(), 6,4));	
			System.out.print("\t"+eva.getTrainTime()+"\n");
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
			}else if(string.equalsIgnoreCase("MBranch")){
				method = SmoothingMethod.MBranch;
			}else if(string.equalsIgnoreCase("HGS_LogLoss")) {
				method = SmoothingMethod.HGS_LogLoss;
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
		
		string = Utils.getOption('S', options);
		if (string.length() != 0) {
			
			if(string.equalsIgnoreCase("Beta")) {
				methodG = GradientMethod.Beta;
			}else if(string.equalsIgnoreCase("L2Norm")){
				methodG = GradientMethod.L2Norm;
			}else if(string.equalsIgnoreCase("EarlyStop")){
				methodG = GradientMethod.EarlyStop;
			}else if(string.equalsIgnoreCase("L2NormAll")) {
				methodG = GradientMethod.L2NormAll;
			}else if(string.equalsIgnoreCase("nonstop")) {
				methodG = GradientMethod.NonStop;
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
