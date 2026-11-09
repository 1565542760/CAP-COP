"""Synthetic final-method checks; no patient data or paper-model training."""
from pathlib import Path
import ast
import json
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'protocols')]
from cap_cop.clinical import clinical_missingness_simulation as preprocessing
from cap_cop.clinical import clinical_transportability_protocol as definitions
from cap_cop.clinical import full_structured_input_strategy_experiment as clinical
from cap_cop.data.model_contracts import align_radiomics_training_inputs
from cap_cop.pipelines.locked_weighted_multimodal_fusion_pipeline import select_weight_from_df, metric_values
import run_isolated_a_roi as roi
import run_isolated_b_comparison as chest
import run_isolated_c_sp6 as stable
import run_gradient as gradient


class FinalContracts(unittest.TestCase):
    def setUp(self):
        self.groups=json.loads((ROOT/'configs/paper/clinical_gradient.json').read_text())['feature_sets']
        self.config=json.loads((ROOT/'configs/paper/final_models.json').read_text(encoding='utf-8'))

    def test_membership_counts(self):
        self.assertEqual([len(self.groups[str(n)]) for n in (7,9,11,16)],[7,9,11,16])

    def test_membership_unique(self):
        for group in self.groups.values():
            self.assertEqual(len(group),len(set(group)))

    def test_nested_sets(self):
        groups=[set(self.groups[str(n)]) for n in (7,9,11,16)]
        self.assertTrue(all(a<b for a,b in zip(groups,groups[1:])))

    def test_eleven_adds_height_and_weight(self):
        self.assertEqual(set(self.groups['11'])-set(self.groups['9']),{'height_cm','weight_kg'})

    def test_runner_uses_same_groups(self):
        for key,n in [('core7',7),('core9_hscrp_pct',9),('transportable11',11),('full16',16)]:
            self.assertEqual(gradient.GRADIENTS[key],self.groups[str(n)])

    def test_full_clinical_members(self):
        config=json.loads((ROOT/'configs/clinical/full_structured_clinical.json').read_text())
        self.assertEqual(config['features'],clinical.FEATURE_SETS['full16'])
        self.assertEqual(config['features'],definitions.ALL_FEATURES)

    def test_positive_class_and_threshold(self):
        self.assertEqual(self.config['positive_class'],'COP')
        self.assertEqual(self.config['classification_threshold'],0.5)
        for fn in (chest.metrics,gradient.metrics):
            m=fn(np.array([0,0,1,1]),np.array([0.1,0.5,0.5,0.9]))
            self.assertEqual(m['Sensitivity'],1.)
            self.assertEqual(m['Specificity'],0.5)

    def test_frozen_weights(self):
        for key,w in [('lesion_roi_radiomics',.75),('full_chest_3d_ct',.65),('stable_multirepresentation_ct',.70)]:
            m=self.config['models'][key]
            self.assertEqual(m['external_clinical_weight'],w)
            self.assertAlmostEqual(w+m['external_imaging_weight'],1.)
            self.assertEqual(len(m['internal_clinical_weights']),5)

    def test_three_final_configurations(self):
        self.assertEqual(len(self.config['models']),3)
        self.assertEqual({m['paper_alias'] for m in self.config['models'].values()},{'A','B','C'})

    def test_final_clinical_rf_seeds(self):
        self.assertEqual(clinical.SEEDS,[20260908,20260909,20260910])

    def test_full_chest_probe(self):
        probe=chest.make_probe()
        model=probe.steps[-1][1]
        self.assertEqual(model.C,1.)
        self.assertEqual(model.solver,'lbfgs')
        self.assertEqual(model.class_weight,'balanced')
        self.assertEqual(model.random_state,20260909)

    def test_cleaner_retains_bmi_rule(self):
        f=pd.DataFrame({'height_cm':[200.,0.], 'weight_kg':[80.,60.],'body_mass_index':[np.nan,0.]})
        out=definitions.FrameCleaner().fit_transform(f)
        self.assertEqual(out.body_mass_index.iloc[0],20.)
        self.assertTrue(np.isnan(out.height_cm.iloc[1]))
        self.assertTrue(np.isnan(out.body_mass_index.iloc[1]))

    def test_imputation_training_only(self):
        train=pd.DataFrame({'age':[20.,40.,np.nan]})
        fitted=preprocessing.FoldImputer(['age'],'median',1).fit(train)
        self.assertEqual(fitted.medians['age'],30.)
        result=fitted.transform(pd.DataFrame({'age':[np.nan,1000.]}))
        self.assertEqual(result.age.iloc[0],30.)
        self.assertEqual(fitted.medians['age'],30.)

    def test_missing_indicators_are_derived(self):
        features=['age','fever']
        original=pd.DataFrame({'age':[20.,np.nan],'fever':[0.,1.]})
        imputed=original.fillna(30.)
        design=preprocessing.make_design(imputed,original,features)
        self.assertEqual(design['__missing_age'].tolist(),[0.,1.])
        self.assertEqual(len(features),2)
        self.assertGreater(len(design.columns),len(features))

    def test_balancing_does_not_change_eval_frame(self):
        frame=pd.DataFrame({'age':[10.,20.,30.,40.],'__missing_age':[0.,0.,0.,0.]})
        backup=frame.copy(deep=True)
        out,y=clinical.augment_training_design(frame,np.array([0,0,0,1]),['age'],'balanced_bootstrap',2)
        self.assertEqual(np.bincount(y).tolist(),[3,3])
        pd.testing.assert_frame_equal(frame,backup)
        self.assertEqual(set(out.age),set(frame.age))

    def test_roi_fit_rejects_patient_overlap(self):
        with self.assertRaises(ValueError):
            roi.fit_a(None,None,None,['synthetic-id'],['synthetic-id'],0,ROOT/'unused')

    def test_gradient_fit_rejects_patient_overlap(self):
        frame=pd.DataFrame({'public_subject_id':['synthetic-id']})
        with self.assertRaises(ValueError):
            gradient.fit_predict(frame,frame,['age'],1,ROOT/'unused')

    def test_patient_alignment_uses_ids(self):
        ids=[f'synthetic-{i:02d}' for i in range(10)]
        labels=pd.DataFrame({'neutral_patient_id':ids,'label_binary':[0,1]*5,'outer_fold':np.repeat(range(5),2)})
        baseline=labels.assign(feature=np.arange(10.)).sample(frac=1.,random_state=11)
        x,y,folds=align_radiomics_training_inputs(labels,baseline,['feature'])
        self.assertEqual(y.tolist(),labels.set_index('neutral_patient_id').loc[x.index,'label_binary'].tolist())
        self.assertEqual(folds.tolist(),baseline.outer_fold.tolist())

    def test_patient_alignment_rejects_duplicate(self):
        labels=pd.DataFrame({'neutral_patient_id':['synthetic-1','synthetic-1'],'label_binary':[0,1]})
        with self.assertRaises(ValueError):
            align_radiomics_training_inputs(labels,labels.assign(feature=[1,2]),['feature'])

    def test_weight_selector_grid(self):
        frame=pd.DataFrame({'label':[0,0,1,1],'p_clinical':[.1,.2,.8,.9],'p_image':[.4,.3,.6,.7]})
        w,audit,grid=select_weight_from_df(frame)
        self.assertEqual(len(grid),21)
        self.assertEqual(grid.weight_clinical.tolist(),[i/20 for i in range(21)])
        self.assertIn(w,grid.weight_clinical.tolist())
        self.assertEqual(audit['selected_weight'],w)

    def test_rank_auroc_with_ties(self):
        self.assertAlmostEqual(metric_values([0,0,1,1],[.1,.5,.5,.9])['AUROC'],.875)

    def test_all_sources_parse(self):
        for folder in ['src','protocols','scripts','tests']:
            for p in (ROOT/folder).rglob('*.py'):
                ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))

    def test_historical_drivers_excluded(self):
        forbidden=['src/cap_cop/models/registry.py','src/cap_cop/clinical/full_structured_model.py',
                   'src/cap_cop/vendor/tangerine_3d_mae/models_mae.py',
                   'src/cap_cop/vendor/tangerine_3d_mae/mains_predict/main_predict.py']
        for rel in forbidden:
            self.assertFalse((ROOT/rel).exists())

    def test_bundled_configs_match_source_configs(self):
        for p in (ROOT/'configs').rglob('*'):
            if p.is_file():
                self.assertEqual(p.read_bytes(),(ROOT/'src/cap_cop/configs'/p.relative_to(ROOT/'configs')).read_bytes())

if __name__=='__main__':
    unittest.main()
