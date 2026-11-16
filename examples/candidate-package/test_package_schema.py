"""Keep the published structural schema aligned with the passive package loader."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from passage_eval.candidates.cli import main
from passage_eval.protocol.candidate_package import FIELDS, load_candidate_package

EXAMPLE = Path(__file__).parent
SCHEMA = EXAMPLE.parents[1] / 'schemas/candidate-package-v1.schema.json'


@unittest.skipUnless(importlib.util.find_spec('jsonschema'), 'jsonschema is optional for example checks')
class CandidateSchemaTests(unittest.TestCase):
    def test_sealed_example_matches_schema_and_exact_loader_fields(self):
        from jsonschema import Draft202012Validator
        schema = json.loads(SCHEMA.read_text())
        Draft202012Validator.check_schema(schema)
        self.assertEqual(set(schema['properties']), FIELDS)
        self.assertEqual(set(schema['required']), FIELDS)
        validator = Draft202012Validator(schema)
        with tempfile.TemporaryDirectory(prefix='passage-example-schema-') as directory:
            root = Path(directory) / 'package'
            shutil.copytree(EXAMPLE, root, ignore=shutil.ignore_patterns('__pycache__', '.venv', 'candidate.json'))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['freeze', str(root/'source.json'), '--out', str(root/'candidate.json')]), 0)
            document = load_candidate_package(root/'candidate.json').document
            validator.validate(document)
            # An unsealed descriptor is deliberately not a sealed manifest.
            self.assertTrue(list(validator.iter_errors(json.loads((root/'source.json').read_text()))))
            for field in FIELDS:
                with self.subTest(missing=field):
                    changed = copy.deepcopy(document); del changed[field]
                    self.assertTrue(list(validator.iter_errors(changed)))
            for name, mutate in [
                ('unknown field', lambda d: d.update(extra=True)),
                ('negative seed', lambda d: d.update(seed=-1)),
                ('duplicate backend', lambda d: d.update(backends=['cpu', 'cpu'])),
                ('unsupported budget', lambda d: d['budget'].update(max_steps=1)),
                ('zero limit', lambda d: d['budget'].update(max_expansions=0)),
                ('unidentified dataset', lambda d: d['dataset'].update(revision='main')),
                ('empty provenance', lambda d: d['training'].update(test_usage='')),
                ('escaping path', lambda d: d['artifacts'][0].update(path='../outside')),
                ('identity newline', lambda d: d.update(candidate_id=d['candidate_id']+'\n')),
                ('entrypoint newline', lambda d: d.update(entrypoint=d['entrypoint']+'\n')),
                ('revision newline', lambda d: d['dataset'].update(revision=d['dataset']['revision']+'\n')),
                ('hash newline', lambda d: d['artifacts'][0].update(sha256=d['artifacts'][0]['sha256']+'\n')),
            ]:
                with self.subTest(invalid=name):
                    changed = copy.deepcopy(document); mutate(changed)
                    self.assertTrue(list(validator.iter_errors(changed)))


if __name__ == '__main__':
    unittest.main()
