import statistics,unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.test02d.sources import ROOT,CANDIDATES,SEEDS,LABELS,load
from src.analysis.test02d.figures import make_figure

class TestFigureValues(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=load(ROOT.parent/'absa-run/validation_aggregate_assumed_equivalence.json')
 def test_model_point_values_match_sources(self):
  fig=make_figure(1,self.data)
  try:
   for axis,key in zip(fig.axes,('joint_macro_f1','joint_micro_f1')):
    markers=[line for line in axis.lines if line.get_marker() in ('o','s','^')]
    self.assertEqual(len(markers),3)
    for line,c in zip(markers,CANDIDATES):
     expected=statistics.mean(float(self.data['model'][c,str(seed)][key]) for seed in SEEDS)
     self.assertAlmostEqual(float(line.get_ydata()[0]),expected)
  finally:plt.close(fig)
 def test_heatmap_includes_all_21_labels_and_values(self):
  fig=make_figure(4,self.data)
  try:
   matrix=fig.axes[0].images[0].get_array();self.assertEqual(matrix.shape,(21,3))
   for i,label in enumerate(LABELS):
    for j,c in enumerate(CANDIDATES):
     expected=statistics.mean(float(self.data['pairs'][c,str(seed),label]['f1']) for seed in SEEDS)
     self.assertAlmostEqual(float(matrix[i,j]),expected)
  finally:plt.close(fig)
if __name__=='__main__':unittest.main()
