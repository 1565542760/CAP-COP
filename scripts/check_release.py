"""Check the source-only publication boundary against the reviewed allowlist."""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
REVIEWED_SHA256 = {'.gitattributes': 'd60f352d0db1404c70afb4bb8b2ca3fd1c610572aa40720e8a0b7baa7885418c', '.github/workflows/final-method-checks.yml': '2cfa70d3983783cab77151addff19e9a0c278e64707207e16b987bed9b6a8094', '.gitignore': '691d2570d6ef3e470a2c63ff73e532f21e42b1117a1a3ebcb1b0f6c2716d9727', 'configs/clinical/full_structured_clinical.json': 'dabf1f38b5bec6aab37ba20c6293f5e2d656660ef0efea9e6a8f56e6a604f202', 'configs/lesion_roi_pyradiomics_settings.yaml': '65ce33c85d2ea88d4c0ccbb2c80d94a0f1fe05e2dfb210f6461f03ccc3980d12', 'configs/lesion_segmentation.yaml': '26eeb0c42e7557bf6f69674875e4942ac87697f9fdcfba2cc322a1db73c8cce4', 'configs/paper/clinical_gradient.json': '89ace49d0ea46ee1e161538dd5155be43873c1ffc326646e9bbf07d15cd30d02', 'configs/paper/final_models.json': '61a5c4700b4c22909eab0f8cb86dbf5a7a67b390d8346c105ad9603e459dfe82', 'protocols/finalize_isolated_a.py': '39a6f00f903e768a0807d3914600a134cdc458fef106edeca5fb01a6f8c579dd', 'protocols/fit_final_a_refit.py': '9b59aaf9e2894f9a4e30749e973629dfad9b7c6ee7cfa7d82cd2afc40dce7c14', 'protocols/run_gradient.py': 'e6257c94d557c0743acfcf6c3c5c129b4f8f709fad5407b94d9a4dac4c6c3f4e', 'protocols/run_isolated_a_roi.py': '9f6e743576a6269970b949d4463295539db2e257c9d93b2aae5c84caa3e33547', 'protocols/run_isolated_b_comparison.py': 'be0682752a82dcacac65086e57bc0c25e10e44dee933453e77ce4beeb08b13e2', 'protocols/run_isolated_c_sp6.py': 'eb34722936bf7b2802dc5af37463c08ad961abb02ec334402034ebdec050980d', 'pyproject.toml': 'e38916aaa08923727024506cb98f966bc370b95586b41efcac1df5ca98e340a3', 'requirements-tabular-locked.txt': '51979a7600199713d184206687e605776d071b937dda551eaf2adb5fce91a0aa', 'scripts/check_release.py': None, 'src/cap_cop/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/clinical/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/clinical/clinical_missingness_simulation.py': '44b3b99ffa1f17626188f819fc34ab8feb7c5f4d36cdf631e6b5ef22618b117c', 'src/cap_cop/clinical/clinical_transportability_protocol.py': 'a7e4e380ca9956d2e6bf9c3af0503cf71ae1c87e54d42c572d9a48607868e457', 'src/cap_cop/clinical/full_structured_input_strategy_experiment.py': '44b7aeac0ca00b48b4f1bab0475e167a57e2891849740738aec6bb59660eb412', 'src/cap_cop/configs/clinical/full_structured_clinical.json': 'dabf1f38b5bec6aab37ba20c6293f5e2d656660ef0efea9e6a8f56e6a604f202', 'src/cap_cop/configs/lesion_roi_pyradiomics_settings.yaml': '65ce33c85d2ea88d4c0ccbb2c80d94a0f1fe05e2dfb210f6461f03ccc3980d12', 'src/cap_cop/configs/lesion_segmentation.yaml': '26eeb0c42e7557bf6f69674875e4942ac87697f9fdcfba2cc322a1db73c8cce4', 'src/cap_cop/configs/paper/clinical_gradient.json': '89ace49d0ea46ee1e161538dd5155be43873c1ffc326646e9bbf07d15cd30d02', 'src/cap_cop/configs/paper/final_models.json': '61a5c4700b4c22909eab0f8cb86dbf5a7a67b390d8346c105ad9603e459dfe82', 'src/cap_cop/data/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/data/model_contracts.py': 'e395e741396096d9ec14bc3fd4e0e1c5d9a328b17c168bb8b457fd4c3d0b6760', 'src/cap_cop/data/release_inputs.py': 'ffe980c110b1ef2f7fff58227cd6a163381f5bf31c87f3ac49199c174500fbf4', 'src/cap_cop/evaluation/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/evaluation/bootstrap.py': '56312e7893e454c808593202d08cc84072bfd9fc2faafda86f41b383a243dcd0', 'src/cap_cop/evaluation/delong.py': '5a6dae31d0e829f7680b4a874719e3a392256e68bfffc95cc9a64ffe0d4acb72', 'src/cap_cop/imaging/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/imaging/foundation/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/imaging/foundation/genesis_encoder.py': 'd401137b44246817c45bcb985d8954ea4b9214d5ce13ebfaa86585a29ca8ed91', 'src/cap_cop/imaging/foundation/tangerine_encoder.py': 'e58e3a92eceaf4ba36e72e9f9e97ad800c45344ea54b8070e56aafd896ba5cf0', 'src/cap_cop/imaging/radiomics/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/imaging/radiomics/lesion_phenotype.py': 'b587d2af120f959f1dc75e519fa4a8371e94fe95cbd586c2623cc93af8acb66a', 'src/cap_cop/imaging/radiomics/mask_contract.py': '802d9e743d572d92e2029f0ae036202e97edd09b42819a40a58cda5e673bceff', 'src/cap_cop/imaging/radiomics/per_slice_dicom_sampler.py': 'ae963516ab07955e04d14a10caf99c98941c92acc69e5b28f72511f9c14367a7', 'src/cap_cop/imaging/radiomics/phenotype_extraction.py': '6911723de5cfaa53c48b24719d7ebbb2c62326f0eb360d9ff08f81cb6ffc8723', 'src/cap_cop/imaging/segmentation/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/imaging/segmentation/inference.py': '879825785fbd0203c720b3df8850adc5054398211263927168424739c0f52f60', 'src/cap_cop/imaging/segmentation/model.py': 'f91c4fbc43743e51eac7b1b23d9258597b142e390c1f972d310953f520ad7677', 'src/cap_cop/imaging/segmentation/predict_external.py': 'fb35ec8ddcb35474692eca315bc7ee748942be19482cfbe9ff88b039d3fdde37', 'src/cap_cop/imaging/segmentation/predict_internal.py': 'e96d1a8c1dda8874d267a0fc68d27b541c48f5e1f03d36567362fe865b2b861b', 'src/cap_cop/imaging/segmentation/preprocess.py': '59fc1555041b136df310ff3b80f869568edf34cf30ec94a59dcb0e659f5c5e7f', 'src/cap_cop/imaging/segmentation/splits.py': '67373adb912c871a998042d75105accd3cac05e83f2fdec7d638cf6b5cb8b9c0', 'src/cap_cop/imaging/whole_chest_tensor/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/imaging/whole_chest_tensor/generate_whole_chest_tensor.py': 'f7e41fc21a5c800d7e205b00191ad88b8e64fe7db79539839aebc2f0bbfe01cd', 'src/cap_cop/imaging/whole_chest_tensor/locked_p95_tensor_contract.py': 'ad9e345e5aaaede68e9f7f4ec22f0f349f1af40b73da5f38bbb9d59e3ed0f8b4', 'src/cap_cop/paths.py': '86a93767c9fae7d787f431d2b8ce39a110ed6b1843dd4871f2fb77e8f0251efe', 'src/cap_cop/pipelines/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/pipelines/lesion_radiomics_replay_pipeline.py': 'cd3cb4c2bd5e2ac51e028f70f8056b6895203ae5eaa99a38900d23768f2b0a2b', 'src/cap_cop/pipelines/locked_weighted_multimodal_fusion_pipeline.py': '2a80f089c191adaac49de2cd3e7f5d1fbd41f3d9778f2b61e17299c86d0bb22b', 'src/cap_cop/pipelines/stable_multirepresentation_replay_pipeline.py': 'b2806aab046900d7f4c551af943b360a19fbecadb1e95dabbb6c6bb434013c55', 'src/cap_cop/vendor/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/vendor/models_genesis/__init__.py': '943df516afa4505b30a914f2196b9d51cf24e243cf8de7b4592eecd977b1999a', 'src/cap_cop/vendor/models_genesis/unet3d.py': '4a660c5403206ee190ac339321f533eb738263cec83e4c9113ef1652ebc17b12', 'src/cap_cop/vendor/tangerine_3d_mae/__init__.py': '128a7c4e737f5831005ccde5b8086262ca7e42a18f4f5aefe8c5b91d00f04662', 'src/cap_cop/vendor/tangerine_3d_mae/mains_predict/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'src/cap_cop/vendor/tangerine_3d_mae/models_vit.py': 'f1da69446ea828633f3d2f66445309a35959c6342c7f48a874807c99c24779ae', 'src/cap_cop/vendor/tangerine_3d_mae/patch_three_d.py': '5325857c8fff1e6879c8b6b918b99342cba943e5b1d34ffea8ba3a4b1bdeae78', 'src/cap_cop/vendor/tangerine_3d_mae/util/__init__.py': 'b67d088796178a5f034b2f16f1a53bde7ff6b982b053f3efd885c0924874d9ff', 'src/cap_cop/vendor/tangerine_3d_mae/util/pos_embed.py': '29108f896817cb4ebb56eca3278602bf3b24ba28912968f2eaff617dbecbab3e', 'tests/test_final_contracts.py': 'b45868520531b7dcff04ebeb586b194cfc2bdaed849ce5f4bc99a09276396c7e'}

SENSITIVE_EXTENSIONS={'.pt','.pth','.ckpt','.pkl','.joblib','.safetensors','.onnx','.dcm','.xlsx','.xls','.npy','.npz','.parquet'}
PATTERNS={
    'machine_user_path':re.compile(r'[A-Za-z]:[/\\]+Users[/\\]+',re.I),
    'wechat_path':re.compile(r'wxid_[a-z0-9_]+',re.I),
    'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

def main():
    entries=REVIEWED_SHA256
    errors=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__','.pytest_cache','build','dist']) or any(x.endswith('.egg-info') for x in p.parts):
            continue
        rel=p.relative_to(ROOT).as_posix()
        if rel=='scripts/check_release.py':
            pass
        if rel not in entries:
            errors.append({'file':rel,'issue':'outside reviewed allowlist'})
        elif entries[rel] is not None and hashlib.sha256(p.read_bytes()).hexdigest()!=entries[rel]:
            errors.append({'file':rel,'issue':'SHA256 changed after review'})
        if p.suffix.lower() in SENSITIVE_EXTENSIONS or rel.startswith(('data/','outputs/')):
            errors.append({'file':rel,'issue':'governed artifact in source tree'})
        try:
            text=p.read_text(encoding='utf-8-sig')
        except UnicodeError:
            errors.append({'file':rel,'issue':'unexpected binary'})
            continue
        for name,pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append({'file':rel,'issue':name})
    for rel in entries:
        if not (ROOT/rel).is_file():
            errors.append({'file':rel,'issue':'allowlisted source missing'})
    print(json.dumps({'checked_files':len(entries),'status':'PASS' if not errors else 'FAIL','issues':errors},indent=2))
    return bool(errors)

if __name__=='__main__':
    sys.exit(main())
