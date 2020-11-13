from django.test import Client, TestCase


class HttpTest(TestCase):
    def test_http_status_codes(self):
        c = Client()
        self.assertEqual(c.get("/").status_code, 302)
        self.assertEqual(c.get("/demo/").status_code, 200)
