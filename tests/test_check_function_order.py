"""The function-order checker (tools/check_function_order.py) flags what the style rule forbids, accepts the rest."""
import importlib.util
import textwrap
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'check_function_order.py'
_spec = importlib.util.spec_from_file_location('check_function_order', TOOL)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def _check(tmp_path, name, text):
    """Run the checker on one file; return (exit status, printed lines)."""
    path = tmp_path / name
    marker = '# ---- ' if name.endswith('.py') else '// ---- '  # fixtures spell a group label as "@@ label @@"
    path.write_text(textwrap.dedent(text).replace('@@ ', marker).replace(' @@', ' ----'), encoding='utf-8')
    problems = []
    try:
        (checker.check_python if name.endswith('.py') else checker.check_rust)(path, problems)
    except SyntaxError as error:
        problems.append(f'syntax error: {error}')
    return (1 if problems else 0), problems


# ---- conditional definition tests ----

def test_functions_under_a_module_level_if_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        if True:
            def zebra():
                pass

            def alpha():
                pass
        ''')
    assert code == 1


def test_functions_under_try_and_else_branches_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        try:
            import fast
        except ImportError:
            def zebra():
                pass

            def alpha():
                pass
        ''')
    assert code == 1


def test_ordered_conditional_branches_pass_and_each_branch_stands_alone(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        if True:
            def alpha():
                pass

            def zebra():
                pass
        else:
            def beta():
                pass
        ''')
    assert code == 0


def test_nested_local_functions_are_not_module_definitions(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        def alpha():
            def zebra():
                pass

            def beta():
                pass
        ''')
    assert code == 0


# ---- exemption tests ----

def test_an_exemption_needs_a_reason(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        # order-exempt:
        def zebra_decorator(fn):
            return fn


        @zebra_decorator
        def alpha():
            pass
        ''')
    assert code == 1


def test_a_decorator_dependency_can_be_exempted_with_a_reason(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        # order-exempt: the decorator is used by alpha below it
        def zebra_decorator(fn):
            return fn


        @zebra_decorator
        def alpha():
            pass
        ''')
    assert code == 0


def test_a_default_argument_dependency_can_be_exempted_with_a_reason(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        def zebra():  # order-exempt: alpha's default calls it at import time
            return 1


        def alpha(value=zebra()):
            return value
        ''')
    assert code == 0


def test_an_unordered_pair_without_an_exemption_still_fails(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        def zebra():
            pass


        def alpha():
            pass
        ''')
    assert code == 1


def test_a_rust_exemption_needs_a_reason_and_works_with_one(tmp_path):
    bad, _ = _check(tmp_path, 'bad.rs', '''
        // order-exempt:
        const fn zebra() -> u8 { 1 }
        const A: u8 = zebra();
        fn alpha() {}
        ''')
    good, _ = _check(tmp_path, 'good.rs', '''
        // order-exempt: the constant below needs it at compile time
        const fn zebra() -> u8 { 1 }
        const A: u8 = zebra();
        fn alpha() {}
        ''')
    assert (bad, good) == (1, 0)


# ---- ordering tests ----

def test_dunders_lead_the_private_names(tmp_path):
    good, _ = _check(tmp_path, 'good.py', '''
        def __init__():
            pass


        def _1_helper():
            pass


        def public():
            pass
        ''')
    bad, _ = _check(tmp_path, 'bad.py', '''
        def _1_helper():
            pass


        def __init__():
            pass
        ''')
    assert (good, bad) == (0, 1)


def test_private_names_are_alphabetical_before_the_public_ones(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        def _b():
            pass


        def _a():
            pass


        def a():
            pass
        ''')
    assert code == 1


def test_a_property_and_its_setter_count_once(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        class Box:
            @property
            def value(self):
                return 1

            @value.setter
            def value(self, new):
                pass

            def zero(self):
                pass
        ''')
    assert code == 0


# ---- rust tests ----

def test_a_rust_function_after_a_braced_use_line_is_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        use std::{fmt, io};

        fn zebra() {}
        fn alpha() {}
        ''')
    assert code == 1


def test_rust_functions_after_an_inner_attribute_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        #![allow(dead_code)]
        fn zebra() {}
        fn alpha() {}
        ''')
    assert code == 1


def test_rust_functions_after_a_leading_block_comment_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        /* a block comment
           over two lines */
        fn zebra() {}
        /* another */ fn alpha() {}
        ''')
    assert code == 1


def test_rust_functions_in_an_inline_module_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        mod inner {
            fn zebra() {}
            fn alpha() {}
        }
        ''')
    assert code == 1


def test_rust_functions_in_a_compound_cfg_test_module_are_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        #[cfg(all(test, feature = "extra"))]
        mod tests {
            fn zebra() {}
            fn alpha() {}
        }
        ''')
    assert code == 1


def test_rust_functions_on_one_line_are_each_checked(tmp_path):
    code, _ = _check(tmp_path, 'lib.rs', '''
        fn zebra() {} fn alpha() {}
        ''')
    assert code == 1


def test_rust_malformed_source_is_reported_not_passed(tmp_path):
    unbalanced, _ = _check(tmp_path, 'a.rs', 'fn alpha() {\n')
    unterminated, _ = _check(tmp_path, 'b.rs', 'fn alpha() { let s = "open; }\n')
    assert (unbalanced, unterminated) == (1, 1)


def test_rust_non_pub_functions_precede_pub_ones(tmp_path):
    good, _ = _check(tmp_path, 'good.rs', '''
        fn zebra() {}
        pub fn alpha() {}
        ''')
    bad, _ = _check(tmp_path, 'bad.rs', '''
        pub fn alpha() {}
        fn zebra() {}
        ''')
    assert (good, bad) == (0, 1)


# ---- test section tests ----

def test_a_test_outside_any_labelled_group_fails(tmp_path):
    code, _ = _check(tmp_path, 'test_x.py', '''
        def test_one():
            pass
        ''')
    assert code == 1


def test_a_helper_after_a_test_fails(tmp_path):
    code, _ = _check(tmp_path, 'test_x.py', '''
        def alpha():
            pass


        @@ a tests @@

        def test_one():
            pass


        def zebra():
            pass
        ''')
    assert code == 1


def test_helpers_then_labelled_groups_pass(tmp_path):
    code, _ = _check(tmp_path, 'test_x.py', '''
        def _helper():
            pass


        def alpha():
            pass


        @@ a tests @@

        def test_two():
            pass


        def test_one():
            pass


        @@ other tests @@

        def test_three():
            pass
        ''')
    assert code == 0


def test_labels_must_be_alphabetical_with_other_tests_last(tmp_path):
    code, _ = _check(tmp_path, 'test_x.py', '''
        @@ other tests @@

        def test_one():
            pass


        @@ a tests @@

        def test_two():
            pass
        ''')
    assert code == 1


def test_rust_tests_need_a_labelled_group_and_follow_the_helpers(tmp_path):
    unlabelled, _ = _check(tmp_path, 'a.rs', '''
        fn helper() {}

        #[test]
        fn one() {}
        ''')
    helper_after, _ = _check(tmp_path, 'b.rs', '''
        @@ a tests @@

        #[test]
        fn one() {}

        fn helper() {}
        ''')
    good, _ = _check(tmp_path, 'c.rs', '''
        fn helper() {}

        @@ a tests @@

        #[test]
        fn one() {}
        ''')
    assert (unlabelled, helper_after, good) == (1, 1, 0)


# ---- other tests ----

def test_case_insensitive_alphabetical_order_is_accepted(tmp_path):
    code, _ = _check(tmp_path, 'mod.py', '''
        def alpha():
            pass


        def Beta():
            pass


        def gamma():
            pass
        ''')
    assert code == 0


def test_the_repository_passes_its_own_checker():
    root = TOOL.parents[1]
    assert checker.main([str(root / part) for part in ('src', 'tools', 'tests', 'rust/crates')]) == 0
