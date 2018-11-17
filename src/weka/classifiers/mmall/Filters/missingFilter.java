package weka.classifiers.mmall.Filters;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;

public class missingFilter {

	/** The manv filter */
	private static MissingAsNewAttValue m_Manv = null;

	/** Rplace missing value filter **/
	private static ReplaceMissingValues m_RMissv = null;

	/**
	 * takes12 arguments:
	 * - the input ARFF file
	 */
	public static void main(String[] args) throws Exception {
		Instances       m_Instances;
		String          filename;

		File folder = new File(args[0]);
		File[] listoffiles = folder.listFiles();

		for (int f = 0; f < listoffiles.length; f++) {
			File path =  listoffiles[f];			
			String p = path.getAbsolutePath();

			//String p = args[0];		
			System.out.println("----------------------------------------------------------------------");
			System.out.println("Processing " + p);

			m_Instances = new Instances(new BufferedReader(new FileReader(p)));
			m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
			m_Instances.deleteWithMissingClass();

			int n = m_Instances.numAttributes() - 1;
			int N = m_Instances.numInstances();
			int c = m_Instances.numClasses();

			// Replace missing values
			int numNumeric = 0;
			int numNominal = 0;
			for (int i = 0; i < n; i++) {
				if (m_Instances.attribute(i).isNominal()) {
					numNominal++;
				} else if (m_Instances.attribute(i).isNumeric()) {
					numNumeric++;
				}
			}

			System.out.println("Total Instance: " + N + ", and Total Attributes: " + n);
			System.out.println("Number of Numeric Attributes: " + numNumeric);
			System.out.println("Number of Nominal Attributes: " + numNominal);
			System.out.println("\n");

			boolean[] isMissing = new boolean[n];
			for (int i = 0; i < n; i++) {
				isMissing[i] = false;
			}

			double[] average = new double[n];
			Arrays.fill(average, 0.0);

			for (int ii = 0; ii < N; ii++) {
				Instance instance = m_Instances.instance(ii);

				for (int i = 0; i < n; i++) {

					if (instance.attribute(i).isNumeric() && !instance.isMissing(i)) {
						double val = 1/(double)N * instance.value(i);
						average[i] +=  val;
					}

					if (instance.isMissing(i)) {
						isMissing[i] = true;
					}
				}
			}

			System.out.println("Attributes missing: ");
			for (int i = 0; i < n; i++) {
				String str = (m_Instances.attribute(i).isNumeric()) ? "Numeric" : "Nominal";
				String strm = (isMissing[i]) ? "Missing = Yes" : "Missing = No";
				System.out.println(m_Instances.attribute(i).name() + ", " + str + ", " + strm);
			}
			System.out.println("\n");

			filename = p.replace("datasets_O", "datasets_M");
			File arffFile = new File(filename);
			PrintWriter arff = new PrintWriter(new BufferedWriter(new FileWriter(arffFile),1000000));

			String str = "@relation " + filename + "\n\n";
			for (int i = 0; i < n; i++) {
				str += "@attribute ";
				str += m_Instances.attribute(i).name();
				str += " ";
				if (m_Instances.attribute(i).isNumeric()) {
					str += "real";
					str += "\n";
				} else if (m_Instances.attribute(i).isNominal()) {
					str += "{";
					for (int ival = 0; ival < m_Instances.attribute(i).numValues(); ival++) {
						str += m_Instances.attribute(i).value(ival).toString();
						if (ival != m_Instances.attribute(i).numValues() - 1) 
							str += ",";
					}
					if (isMissing[i]) {
						str += ",missing";
					} 
					str += "} \n";
				}
			}

			str += "@attribute class {";
			for (int cval = 0; cval < c; cval++) {
				str += m_Instances.classAttribute().value(cval).toString();
				if (cval != c - 1) 
					str += ",";
			}
			str += "} \n\n";
			str += "@data \n";

			arff.println(str);
			System.out.println(str);

			for (int ii = 0; ii < N; ii++) {
				Instance instance = m_Instances.instance(ii);

				for (int i = 0; i < n; i++) {

					if (instance.attribute(i).isNumeric() && !instance.isMissing(i)) {
						arff.print(instance.value(i) + ",");
					} else if (instance.attribute(i).isNumeric() && instance.isMissing(i)) {	
						arff.print(average[i] + ",");
					} else if (instance.attribute(i).isNominal() && !instance.isMissing(i)) {
						arff.print(instance.attribute(i).value((int)instance.value(i)).toString() + ",");
					} else if (instance.attribute(i).isNominal() && instance.isMissing(i)) {	
						arff.print("missing,");
					}
				}

				int i = instance.classIndex();
				arff.print(instance.attribute(i).value((int)instance.value(i)).toString());
				arff.print("\n");
			}


			arff.flush();
			arff.close();
			System.out.println("File written to "+arffFile.getAbsolutePath());
			System.out.println("----------------------------------------------------------------------");
		}

	}
}

