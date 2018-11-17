package weka.classifiers.mmall.Online.Bayes;

public class BayesNodePYP extends BayesNode {
	protected int[][] xyCountsPYP; //simulated count for the PYP smoothing
	
	public BayesNodePYP(int attNumber, BayesNode parent, BayesNode root, BayesTree forest) {
		super(attNumber, parent, root, forest);
	}
	
	
	@Override
	public void computeFinalProbabilities() {
		// TODO Auto-generated method stub
		super.computeFinalProbabilities();
	}
	

}
