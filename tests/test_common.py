"""Connection settings tests: stage, Secrets Manager and overrides (no AWS call, no database).

Run:  python -m unittest discover tests
"""
import os
import unittest
from unittest import mock

from id_fraud_detector import common

SECRET = {"host": "proxy.example.internal", "port": 5432, "username": "app_user", "password": "not-a-real-password"}
CLEAN = {k: v for k, v in os.environ.items()
         if k not in ("STAGE", "DB_SECRET_NAME", "DB_HOST", "DB_PORT", "DB_NAME", "ANON_SALT")}


def settings(env: dict, secret: dict = SECRET):
    common.db_settings.cache_clear()
    with mock.patch.dict(os.environ, {**CLEAN, **env}, clear=True), \
            mock.patch.object(common, "_fetch_secret", return_value=dict(secret)) as fetch, \
            mock.patch("builtins.print"):
        try:
            return common.db_settings(), fetch
        finally:
            common.db_settings.cache_clear()


class Stage(unittest.TestCase):
    def test_database_name_follows_the_stage(self):
        for value, database in (("dev", "sisteya_dev"), ("live", "sisteya_live"), (" LIVE ", "sisteya_live")):
            result, _ = settings({"STAGE": value})
            self.assertEqual(result["dbname"], database)

    def test_stage_is_required(self):
        for env in ({}, {"STAGE": ""}, {"STAGE": "live; drop"}):
            with self.assertRaises(RuntimeError):
                settings(env)

    def test_tenants_follow_the_stage(self):
        for value, tenants in (("dev", ["demo"]), ("live", ["esmio", "llevaloya"]), ("test", ["esmio", "llevaloya"])):
            with mock.patch.dict(os.environ, {**CLEAN, "STAGE": value}, clear=True):
                self.assertEqual(common.load_config()["tenants"], tenants, value)

    def test_outputs_go_to_a_folder_per_stage(self):
        with mock.patch.dict(os.environ, {**CLEAN, "STAGE": "dev"}, clear=True):
            self.assertEqual(common.output_dir({"output_dir": "data"}), common.ROOT / "data" / "dev")


class Secret(unittest.TestCase):
    def test_credentials_come_from_the_default_secret(self):
        result, fetch = settings({"STAGE": "dev"})
        fetch.assert_called_once_with("sisteya_db_proxy_credentials")
        self.assertEqual(result, {"host": "proxy.example.internal", "port": "5432", "dbname": "sisteya_dev",
                                  "user": "app_user", "password": "not-a-real-password"})

    def test_overrides(self):
        result, fetch = settings({"STAGE": "dev", "DB_SECRET_NAME": "other_secret", "DB_HOST": "localhost",
                                  "DB_PORT": "6543", "DB_NAME": "sisteya_copy"})
        fetch.assert_called_once_with("other_secret")
        self.assertEqual((result["host"], result["port"], result["dbname"]), ("localhost", "6543", "sisteya_copy"))
        self.assertEqual(result["user"], "app_user")

    def test_incomplete_secret_is_rejected(self):
        with self.assertRaises(RuntimeError):
            settings({"STAGE": "dev"}, secret={"host": "h", "username": "u"})

    def test_expired_sso_session_gives_a_clear_message(self):
        from botocore.exceptions import TokenRetrievalError

        session = mock.Mock()
        session.client.return_value.get_secret_value.side_effect = TokenRetrievalError(provider="sso", error_msg="expired")
        with mock.patch.object(common, "aws_session", return_value=session), \
                mock.patch.dict(os.environ, CLEAN, clear=True):
            with self.assertRaisesRegex(RuntimeError, "aws sso login"):
                common._fetch_secret("any")


if __name__ == "__main__":
    unittest.main()
