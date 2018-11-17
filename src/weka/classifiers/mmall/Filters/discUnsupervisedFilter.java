package weka.classifiers.mmall.Filters;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class discUnsupervisedFilter {

	private static weka.filters.unsupervised.attribute.Discretize m_Disc = null;

	/**
	 * takes 2 arguments:
	 * - the input ARFF file
	 * - the attribute index (starting with 1)
	 */
	public static void main(String[] args) throws Exception {
		Instances       m_Instances;
		Instances       m_DiscreteInstances;
		
		ArffSaver       saver;
		String          filename;

		File folder = new File(args[0]);
		File[] listoffiles = folder.listFiles();
		
		int numBins = 100;
		
		//for (int i = 0; i < listoffiles.length; i++) {
			//File path =  listoffiles[i];			
			//String p = path.getAbsolutePath();

		
			String p = args[0];		
			System.out.println("Processing " + p);
			
			m_Instances = new Instances(new BufferedReader(new FileReader(p)));
			m_Instances.setClassIndex(m_Instances.numAttributes() - 1);
			m_Instances.deleteWithMissingClass();

			// Discretize instances if required
			m_Disc = new weka.filters.unsupervised.attribute.Discretize();
			m_Disc.setUseEqualFrequency(true);
			m_Disc.setBins(numBins);
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(m_Instances);
			System.out.println("Applying Filter");
			m_DiscreteInstances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
			System.out.println("Done");

			//filename = p.replace("datasets_originals", "dataset_DM");
			filename = p.replace("datasets_O", "datasets_DM2");
			
			saver = new ArffSaver();
			
			saver.setInstances(m_DiscreteInstances);
			
			saver.setFile(new File(filename));
			saver.setDestination(new File(filename));
			saver.writeBatch();
			
			System.out.println("File Writen - " + filename);		
			System.out.println("----------------------------------------------------------------");
		//}

	}
}

