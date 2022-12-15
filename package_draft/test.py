import unittest
from BeadFinder import BeadFinder
from helper import coor_tuple_to_int
import tempfile
import os


class BeadFinderTestCase(unittest.TestCase):
    """test BeadFinder"""

    def test_list_of_well_ids(self):
        """test if list of well ids calculate successfully"""
        image_path = 'src.jpeg'
        algorithm_name = ['adaptive_thresholding', 'average_thresholding', 'deep_learning', 'deep_learning_aug']
        label = 'test_img'

        ids_avg = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A15', 'A16', 'A17', 'A18', 'A19',
                   'A20', 'A22', 'A24', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B13',
                   'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'C2', 'C4', 'C6', 'C8',
                   'C10', 'C12', 'C14', 'C16', 'C18', 'C20', 'C24', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                   'D9', 'D10', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D19', 'D22', 'D23', 'E2', 'E4', 'E6', 'E8',
                   'E10', 'E12', 'E14', 'E16', 'E18', 'E20', 'E22', 'F1', 'F3', 'F5', 'F7', 'F9', 'F11', 'F13', 'F15',
                   'F17', 'F19', 'F21', 'F23', 'G2', 'G4', 'H1', 'H3', 'H5', 'H7', 'H9', 'H11', 'H13', 'H15', 'H17',
                   'H19', 'H21', 'H23', 'J1', 'J3', 'J5', 'J7', 'J9', 'J11', 'J13', 'J15', 'J17', 'J21', 'J23', 'L1',
                   'L4', 'L5', 'L7', 'L9', 'L11', 'L13', 'L15', 'L17', 'L19', 'L21', 'L23', 'N3', 'N5', 'P11', 'P14']
        ids_adap = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A14', 'A15', 'A16', 'A17',
                    'A18', 'A19', 'A20', 'A22', 'A24', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
                    'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B22', 'B23', 'B24', 'C2',
                    'C4', 'C6', 'C8', 'C10', 'C12', 'C14', 'C16', 'C18', 'C20', 'C22', 'C24', 'D1', 'D2', 'D3', 'D4',
                    'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D19', 'D21',
                    'D23', 'E2', 'E4', 'E6', 'E8', 'E10', 'E12', 'E14', 'E16', 'E18', 'E20', 'E22', 'E24', 'F1', 'F3',
                    'F5', 'F7', 'F9', 'F11', 'F13', 'F15', 'F17', 'F19', 'F21', 'F23', 'G2', 'G4', 'H1', 'H3', 'H5',
                    'H7', 'H9', 'H11', 'H13', 'H15', 'H17', 'H19', 'H21', 'H23', 'J1', 'J3', 'J5', 'J7', 'J9', 'J11',
                    'J13', 'J15', 'J17', 'J19', 'J21', 'J23', 'L1', 'L3', 'L5', 'L7', 'L9', 'L11', 'L13', 'L15', 'L17',
                    'L19', 'L21', 'L23', 'N3', 'N5', 'P11']
        ids_dl = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',
                  'A17', 'A18', 'A19', 'A20', 'A22', 'A24', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24',
                  'C2', 'C4', 'C6', 'C8', 'C10', 'C12', 'C14', 'C16', 'C18', 'C20', 'C21', 'C22', 'C24', 'D1', 'D2',
                  'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17',
                  'D19', 'D21', 'D23', 'E2', 'E4', 'E6', 'E8', 'E10', 'E12', 'E14', 'E15', 'E16', 'E18', 'E20', 'E22',
                  'E24', 'F1', 'F3', 'F5', 'F7', 'F9', 'F11', 'F13', 'F15', 'F17', 'F19', 'F21', 'F23', 'G2', 'G4',
                  'H1', 'H2', 'H3', 'H5', 'H7', 'H9', 'H11', 'H13', 'H15', 'H17', 'H19', 'H21', 'H23', 'I13', 'J1',
                  'J3', 'J5', 'J7', 'J9', 'J11', 'J13', 'J15', 'J17', 'J19', 'J21', 'J23', 'L1', 'L3', 'L4', 'L5', 'L7',
                  'L9', 'L11', 'L13', 'L15', 'L17', 'L19', 'L21', 'L23', 'M3', 'M4', 'N3', 'N5', 'P12', 'P13', 'P18']
        ids_dl_aug = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                      'A16', 'A17',
                      'A18', 'A19', 'A20', 'A22', 'A24', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
                      'B11', 'B12',
                      'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'C2', 'C4',
                      'C6', 'C8',
                      'C10', 'C12', 'C14', 'C16', 'C18', 'C20', 'C21', 'C22', 'C24', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                      'D7', 'D8',
                      'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D19', 'D21', 'D23', 'E2', 'E4',
                      'E6', 'E8',
                      'E10', 'E12', 'E14', 'E16', 'E18', 'E20', 'E22', 'E24', 'F1', 'F3', 'F5', 'F7', 'F9', 'F11',
                      'F13', 'F15',
                      'F17', 'F19', 'F21', 'F23', 'G2', 'G4', 'H1', 'H2', 'H3', 'H5', 'H7', 'H9', 'H11', 'H13', 'H15',
                      'H17', 'H19',
                      'H21', 'H23', 'J1', 'J3', 'J5', 'J7', 'J9', 'J11', 'J13', 'J15', 'J17', 'J19', 'J21', 'J23', 'L1',
                      'L3', 'L5',
                      'L7', 'L9', 'L11', 'L13', 'L15', 'L17', 'L19', 'L21', 'L23', 'M3', 'M4', 'N3', 'N5', 'P12', 'P13']
        ids_gt = [ids_adap, ids_avg, ids_dl, ids_dl_aug]

        for i in range(len(algorithm_name)):
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = os.path.join(temp_dir.name, label + '.png')
            instance = BeadFinder(image_path, algorithm_name[i], label, False,
                                  False, output_dir)
            beads_ids, _, _ = instance.find_beads()
            self.assertEqual(beads_ids, ids_gt[i])
            temp_dir.cleanup()

    def test_list_of_row_wise_coors(self):
        """test if list of row wise coors calculate successfully"""
        image_path = 'src.jpeg'
        algorithm_name = ['adaptive_thresholding', 'average_thresholding', 'deep_learning', 'deep_learning_aug']
        label = 'test_img'

        coords_avg = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
                      21, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 71, 3,
                      7, 11, 15, 19, 23, 27, 31, 35, 39, 47, 51, 55, 59, 63, 67, 75, 87, 91, 9, 19, 29, 39, 49, 59, 69,
                      79, 89, 99, 109, 5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137, 13, 27, 7, 23, 39, 55, 71, 87,
                      103, 119, 135, 151, 167, 183, 9, 29, 49, 69, 89, 109, 129, 149, 169, 209, 229, 11, 47, 59, 83,
                      107, 131, 155, 179, 203, 227, 251, 275, 41, 69, 175, 223]
        coords_adap = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
            23, 25, 27, 29, 31, 33, 35, 37, 39, 43, 45, 47, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 3, 7, 11, 15,
            19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 75, 83, 91, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109,
            119, 5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137, 13, 27, 7, 23, 39, 55, 71, 87, 103, 119, 135, 151,
            167,
            183, 9, 29, 49, 69, 89, 109, 129, 149, 169, 189, 209, 229, 11, 35, 59, 83, 107, 131, 155, 179, 203, 227,
            251,
            275, 41, 69, 175]
        coors_dl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13,
                    15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 5, 11, 17, 23, 29, 35, 41, 47,
                    53, 59, 62, 65, 71, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 75, 83, 91, 9,
                    19, 29, 39, 49, 59, 69, 74, 79, 89, 99, 109, 119, 5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137,
                    13, 27, 7, 15, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183, 116, 9, 29, 49, 69, 89, 109, 129,
                    149, 169, 189, 209, 229, 11, 35, 47, 59, 83, 107, 131, 155, 179, 203, 227, 251, 275, 38, 51, 41, 69,
                    191, 207, 287]
        coords_dl_aug = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 1, 3, 5, 7, 9,
                         11, 13, 15, 17,
                         19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 5, 11, 17, 23, 29, 35, 41, 47, 53,
                         59, 62, 65, 71,
                         3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 75, 83, 91, 9, 19, 29, 39,
                         49, 59, 69, 79,
                         89, 99, 109, 119, 5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137, 13, 27, 7, 15, 23, 39, 55,
                         71, 87, 103,
                         119, 135, 151, 167, 183, 9, 29, 49, 69, 89, 109, 129, 149, 169, 189, 209, 229, 11, 35, 59, 83,
                         107, 131, 155,
                         179, 203, 227, 251, 275, 38, 51, 41, 69, 191, 207]
        coords_gt = [coords_adap, coords_avg, coors_dl, coords_dl_aug]

        for i in range(len(algorithm_name)):
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = os.path.join(temp_dir.name, label + '.png')
            instance = BeadFinder(image_path, algorithm_name[i], label, False,
                                  False, output_dir)
            _, beads_coors, _ = instance.find_beads()
            beads_coors_to_int = coor_tuple_to_int(beads_coors)
            self.assertEqual(beads_coors_to_int, coords_gt[i])
            temp_dir.cleanup()

    def test_count(self):
        """test if count of meg.beads calculate successfully"""
        image_path = 'src.jpeg'
        algorithm_name = ['adaptive_thresholding', 'average_thresholding', 'deep_learning', 'deep_learning_aug']
        label = 'test_img'

        count_gt = [140, 135, 152, 148]

        for i in range(len(algorithm_name)):
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = os.path.join(temp_dir.name, label + '.png')
            instance = BeadFinder(image_path, algorithm_name[i], label, False,
                                  False, output_dir)
            _, _, count = instance.find_beads()
            self.assertEqual(count, count_gt[i])
            temp_dir.cleanup()

    def test_heatmap_path_exist(self):
        """test if heatmap saved after run"""
        image_path = 'src.jpeg'
        algorithm_name = ['adaptive_thresholding', 'average_thresholding', 'deep_learning', 'deep_learning_aug']
        label = 'test_img'

        for algo_name in algorithm_name:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = os.path.join(temp_dir.name, label+'.png')
            instance = BeadFinder(image_path, algo_name, label, False,
                                  False, output_dir)
            instance.find_beads()
            self.assertTrue(os.path.exists(output_dir))
            temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()