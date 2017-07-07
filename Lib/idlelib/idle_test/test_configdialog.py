"""Test idlelib.configdialog.

Half the class creates dialog, half works with user customizations.
Coverage: 46% just by creating dialog, 56% with current tests.
"""
from idlelib.configdialog import ConfigDialog, idleConf, changes
from test.support import requires
requires('gui')
from tkinter import Tk
import unittest
import idlelib.config as config

# Tests should not depend on fortuitous user configurations.
# They must not affect actual user .cfg files.
# Use solution from test_config: empty parsers with no filename.
usercfg = idleConf.userCfg
testcfg = {
    'main': config.IdleUserConfParser(''),
    'highlight': config.IdleUserConfParser(''),
    'keys': config.IdleUserConfParser(''),
    'extensions': config.IdleUserConfParser(''),
}

root = None
configure = None
mainpage = changes['main']
highpage = changes['highlight']
keyspage = changes['keys']

class TestDialog(ConfigDialog): pass  # Delete?
 

def setUpModule():
    global root, configure
    idleConf.userCfg = testcfg
    root = Tk()
    root.withdraw()
    configure = TestDialog(root, 'Test', _utest=True)


def tearDownModule():
    global root, configure
    idleConf.userCfg = usercfg
    configure.remove_var_callbacks()
    del configure
    root.update_idletasks()
    root.destroy()
    del root


class FontTabTest(unittest.TestCase):

    def setUp(self):
        changes.clear()

    def test_font(self):
        # Set values guaranteed not to be defaults.
        default_font = idleConf.GetFont(root, 'main', 'EditorWindow')
        default_size = str(default_font[1])
        default_bold = default_font[2] == 'bold'
        configure.font_name.set('Test Font')
        expected = {'EditorWindow': {'font': 'Test Font',
                                     'font-size': default_size,
                                     'font-bold': str(default_bold)}}
        self.assertEqual(mainpage, expected)
        changes.clear()
        configure.font_size.set(20)
        expected = {'EditorWindow': {'font': 'Test Font',
                                     'font-size': '20',
                                     'font-bold': str(default_bold)}}
        self.assertEqual(mainpage, expected)
        changes.clear()
        configure.font_bold.set(not default_bold)
        expected = {'EditorWindow': {'font': 'Test Font',
                                     'font-size': '20',
                                     'font-bold': str(not default_bold)}}
        self.assertEqual(mainpage, expected)

    #def test_sample(self): pass  # TODO

    def test_tabspace(self):
        configure.space_num.set(6)
        self.assertEqual(mainpage, {'Indent': {'num-spaces': '6'}})


class HighlightTest(unittest.TestCase):

    def setUp(self):
        changes.clear()

    #def test_colorchoose(self): pass  # TODO


class KeysTest(unittest.TestCase):

    def setUp(self):
        changes.clear()


class GeneralTest(unittest.TestCase):

    def setUp(self):
        changes.clear()

    def test_startup(self):
        configure.radio_startup_edit.invoke()
        self.assertEqual(mainpage,
                         {'General': {'editor-on-startup': '1'}})

    def test_autosave(self):
        configure.radio_save_auto.invoke()
        self.assertEqual(mainpage, {'General': {'autosave': '1'}})

    def test_editor_size(self):
        configure.entry_win_height.insert(0, '1')
        self.assertEqual(mainpage, {'EditorWindow': {'height': '140'}})
        changes.clear()
        configure.entry_win_width.insert(0, '1')
        self.assertEqual(mainpage, {'EditorWindow': {'width': '180'}})

    #def test_help_sources(self): pass  # TODO


if __name__ == '__main__':
    unittest.main(verbosity=2)
