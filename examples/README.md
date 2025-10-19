# Examples

This directory contains working examples and demo code for using QuizGeneration.

## Files

### [web_ui_integration_example.py](web_ui_integration_example.py)

Complete working example demonstrating how to integrate QuizGeneration with a web-based grading UI.

**Shows:**
- How to import and use `grade_from_qr` module
- Direct API: `regenerate_from_metadata()` function
- Full flow: QR scan → decode → regenerate answers
- Answer formatting for web display
- Error handling

**Run it:**
```bash
# Set encryption key
export QUIZ_ENCRYPTION_KEY="your-key-here"

# Run example
python examples/web_ui_integration_example.py
```

**Expected output:**
```
======================================================================
WEB UI INTEGRATION EXAMPLE
======================================================================

2. Direct API: Metadata -> Answers
----------------------------------------------------------------------
Input: type=VirtualAddressParts, seed=12345, version=1.0, points=2.0

Regenerated 1 answer(s):
  • answer: 28

======================================================================
INTEGRATION SUMMARY:
  1. Install this repo alongside your web UI
  2. Import: from grade_from_qr import regenerate_from_metadata
  3. Set QUIZ_ENCRYPTION_KEY environment variable
  4. Decode QR -> Extract metadata -> Call regenerate_from_metadata()
  5. Display answers in your grading UI
======================================================================
```

---

## See Also

- [Web UI Integration Guide](../documentation/WEB_UI_INTEGRATION.md) - Complete API documentation
- [Grading Guide](../documentation/GRADING_GUIDE.md) - CLI-based grading workflows
