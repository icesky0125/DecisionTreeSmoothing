package hdp.testing;

import hdp.ProbabilityTree;
import java.util.Arrays;

public class Test2LevelsExampleHeartAttack {

	
	public static void main(String... args) throws NoSuchFieldException, IllegalAccessException {
		
		String [][]data = {
				{"yes","heavy","tall"},
				{"no","light","short"},
				{"no","heavy","med"},
				{"no","light","med"},
				{"no","light","short"},
				{"no","heavy","med"},
				{"yes","heavy","short"},
				{"no","heavy","tall"},
				{"no","heavy","tall"},
				{"no","light","short"},
				{"no","heavy","tall"},
				{"no","light","med"},
				{"no","heavy","med"},
				{"no","light","med"},
				{"yes","heavy","med"}
		};
		
		ProbabilityTree hdp = new ProbabilityTree();
		hdp.addDataset(data);//learn distribution from the dataset
		
		System.out.println("\t\t\theart attack:\t\t"+Arrays.toString(hdp.getValuesTarget()));
		System.out.println("p(heart attack | heavy & short)=\t"+Arrays.toString(hdp.query("heavy","short")));
		System.out.println("p(heart attack | heavy & tall) =\t"+Arrays.toString(hdp.query("heavy","tall")));
		System.out.println("p(heart attack | light & tall) =\t"+Arrays.toString(hdp.query("light","tall")));
		
		System.out.println("Learnt tree looks like...");
		System.out.println(hdp.printProbabilities());
		
		
	}

}
