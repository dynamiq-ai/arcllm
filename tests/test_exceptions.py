"""
Tests for arcllm.exceptions module.
"""

from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    BudgetExceededError,
    ContentFilterError,
    InternalServerError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    ServiceUnavailableError,
    TimeoutError,
    UnsupportedModelError,
    UnsupportedParameterError,
    map_status_code_to_exception,
)


class TestArcLLMError:
    """Tests for base exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ArcLLMError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"

    def test_error_with_metadata(self):
        """Test error with metadata."""
        error = ArcLLMError(
            "API error", provider="openai", model="gpt-4", status_code=500, request_id="req-123"
        )
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.status_code == 500
        assert error.request_id == "req-123"

    def test_str_includes_metadata(self):
        """Test string representation includes metadata."""
        error = ArcLLMError("API error", provider="openai", status_code=500)
        error_str = str(error)
        assert "API error" in error_str
        assert "provider=openai" in error_str
        assert "status_code=500" in error_str

    def test_repr(self):
        """Test repr includes all fields."""
        error = ArcLLMError("Test", provider="openai", model="gpt-4")
        repr_str = repr(error)
        assert "ArcLLMError" in repr_str
        assert "openai" in repr_str


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error(self):
        """Test authentication error creation."""
        error = AuthenticationError("Invalid API key", provider="openai", status_code=401)
        assert isinstance(error, ArcLLMError)
        assert error.status_code == 401


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error(self):
        """Test rate limit error creation."""
        error = RateLimitError(
            "Rate limit exceeded", retry_after=30.0, provider="openai", status_code=429
        )
        assert isinstance(error, ArcLLMError)
        assert error.retry_after == 30.0


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error(self):
        """Test timeout error creation."""
        error = TimeoutError(
            "Request timed out", timeout_type="read", timeout_seconds=30.0, provider="openai"
        )
        assert isinstance(error, ArcLLMError)
        assert error.timeout_type == "read"
        assert error.timeout_seconds == 30.0


class TestProviderAPIError:
    """Tests for ProviderAPIError."""

    def test_provider_api_error(self):
        """Test provider API error creation."""
        error = ProviderAPIError(
            "Server error",
            error_type="server_error",
            error_code="500",
            provider="openai",
            status_code=500,
        )
        assert isinstance(error, ArcLLMError)
        assert error.error_type == "server_error"
        assert error.error_code == "500"


class TestUnsupportedModelError:
    """Tests for UnsupportedModelError."""

    def test_unsupported_model_error(self):
        """Test unsupported model error creation."""
        error = UnsupportedModelError(
            "Model not found", model="gpt-5-turbo", provider="openai", status_code=404
        )
        assert isinstance(error, ArcLLMError)
        assert error.model == "gpt-5-turbo"


class TestUnsupportedParameterError:
    """Tests for UnsupportedParameterError."""

    def test_unsupported_parameter_error(self):
        """Test unsupported parameter error creation."""
        error = UnsupportedParameterError(
            "Unsupported parameters", unsupported_params=["foo", "bar"], provider="anthropic"
        )
        assert isinstance(error, ArcLLMError)
        assert error.unsupported_params == ["foo", "bar"]


class TestResponseParseError:
    """Tests for ResponseParseError."""

    def test_response_parse_error(self):
        """Test response parse error creation."""
        error = ResponseParseError("Invalid JSON", raw_data=b"not json", provider="openai")
        assert isinstance(error, ArcLLMError)
        assert error.raw_data == b"not json"


class TestContentFilterError:
    """Tests for ContentFilterError."""

    def test_content_filter_error(self):
        """Test content filter error creation."""
        error = ContentFilterError("Content blocked", filter_reason="violence", provider="openai")
        assert isinstance(error, ArcLLMError)
        assert error.filter_reason == "violence"


class TestInvalidRequestError:
    """Tests for InvalidRequestError."""

    def test_invalid_request_error(self):
        """Test invalid request error creation."""
        error = InvalidRequestError(
            "Invalid parameter value", param="temperature", provider="openai", status_code=400
        )
        assert isinstance(error, ArcLLMError)
        assert error.param == "temperature"


class TestMapStatusCode:
    """Tests for map_status_code_to_exception helper."""

    def test_map_401_to_auth_error(self):
        """Test 401 maps to AuthenticationError."""
        error = map_status_code_to_exception(401, "Unauthorized")
        assert isinstance(error, AuthenticationError)

    def test_map_403_to_auth_error(self):
        """Test 403 maps to AuthenticationError."""
        error = map_status_code_to_exception(403, "Forbidden")
        assert isinstance(error, AuthenticationError)

    def test_map_429_to_rate_limit(self):
        """Test 429 maps to RateLimitError."""
        error = map_status_code_to_exception(429, "Too Many Requests")
        assert isinstance(error, RateLimitError)

    def test_map_404_to_unsupported_model(self):
        """Test 404 maps to UnsupportedModelError."""
        error = map_status_code_to_exception(404, "Not Found")
        assert isinstance(error, UnsupportedModelError)

    def test_map_400_to_invalid_request(self):
        """Test 400 maps to InvalidRequestError."""
        error = map_status_code_to_exception(400, "Bad Request")
        assert isinstance(error, InvalidRequestError)

    def test_map_408_to_timeout(self):
        """Test 408 maps to TimeoutError."""
        error = map_status_code_to_exception(408, "Request Timeout")
        assert isinstance(error, TimeoutError)

    def test_map_500_to_internal_server_error(self):
        """500 → InternalServerError (subclass of ProviderAPIError)."""
        error = map_status_code_to_exception(500, "Server Error")
        assert isinstance(error, InternalServerError)
        assert isinstance(error, ProviderAPIError)  # back-compat catch

    def test_map_503_to_service_unavailable(self):
        """503 gets its own subclass for retry-with-backoff classification."""
        error = map_status_code_to_exception(503, "Service Unavailable")
        assert isinstance(error, ServiceUnavailableError)
        assert isinstance(error, ProviderAPIError)

    def test_map_402_to_budget_exceeded(self):
        """402 (Payment Required) maps to BudgetExceededError."""
        error = map_status_code_to_exception(402, "Payment Required")
        assert isinstance(error, BudgetExceededError)
        assert isinstance(error, ProviderAPIError)

    def test_map_429_with_quota_message_to_budget_exceeded(self):
        """429 with quota / billing wording is a budget issue, not a rate limit.

        OpenAI conflates the two on 429 — disambiguating here lets callers
        decide between back-off (RateLimitError) and stop-and-bill
        (BudgetExceededError).
        """
        error = map_status_code_to_exception(
            429, "You exceeded your current quota, please check your plan and billing"
        )
        assert isinstance(error, BudgetExceededError)
        assert not isinstance(error, RateLimitError)

    def test_map_429_without_quota_message_stays_rate_limit(self):
        error = map_status_code_to_exception(429, "Rate limit exceeded; retry in 30s")
        assert isinstance(error, RateLimitError)
        assert not isinstance(error, BudgetExceededError)

    def test_map_unknown_to_provider_error(self):
        """Unknown status maps to ProviderAPIError."""
        error = map_status_code_to_exception(418, "I'm a teapot")
        assert isinstance(error, ProviderAPIError)


class TestNewExceptionClasses:
    """Direct construction smoke for the new subclasses."""

    def test_budget_exceeded_carries_metadata(self):
        err = BudgetExceededError("quota exhausted", provider="openai", status_code=429)
        assert err.provider == "openai"
        assert err.status_code == 429
        assert isinstance(err, ProviderAPIError)

    def test_service_unavailable_inherits_provider_api(self):
        err = ServiceUnavailableError("region down", provider="bedrock", status_code=503)
        assert err.provider == "bedrock"
        assert isinstance(err, ProviderAPIError)

    def test_internal_server_inherits_provider_api(self):
        err = InternalServerError("oops", provider="anthropic", status_code=500)
        assert err.provider == "anthropic"
        assert isinstance(err, ProviderAPIError)


class TestLitellmCompatAliases:
    """Litellm-compat aliases let downstream callers (notably dynamiq) import
    the legacy litellm.exceptions names without code changes. The aliases must
    be the *same class object* as their arcllm-canonical counterparts so that
    issubclass/isinstance checks across the boundary keep working.
    """

    def test_timeout_alias(self):
        from arcllm.exceptions import Timeout
        from arcllm.exceptions import TimeoutError as ArcLLMTimeoutError

        assert Timeout is ArcLLMTimeoutError

    def test_api_connection_error_alias(self):
        from arcllm.exceptions import APIConnectionError
        from arcllm.exceptions import ConnectionError as ArcLLMConnectionError

        assert APIConnectionError is ArcLLMConnectionError

    def test_api_error_alias(self):
        from arcllm.exceptions import APIError, ProviderAPIError

        assert APIError is ProviderAPIError

    def test_bad_request_error_alias(self):
        from arcllm.exceptions import BadRequestError, InvalidRequestError

        assert BadRequestError is InvalidRequestError

    def test_aliases_are_top_level_arcllm_exports(self):
        """dynamiq imports e.g. ``from arcllm import APIError`` — these must
        be reachable from the top-level arcllm namespace too, not just
        ``arcllm.exceptions``."""
        import arcllm

        assert arcllm.APIError is arcllm.exceptions.ProviderAPIError
        assert arcllm.BadRequestError is arcllm.exceptions.InvalidRequestError
        assert arcllm.Timeout is arcllm.exceptions.TimeoutError
        assert arcllm.APIConnectionError is arcllm.exceptions.ConnectionError
