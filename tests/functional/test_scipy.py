import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
from morphocore.functional import dilation, erosion

def create_simple_test_cases():
    """Crée des cas de test simples pour déboguer"""
    cases = {}
    
    # Cas 1: Image très simple - un seul pixel
    img1 = np.zeros((7, 7), dtype=np.float32)
    img1[3, 3] = 1.0
    cases['single_pixel'] = img1
    
    # Cas 2: Petit carré
    img2 = np.zeros((9, 9), dtype=np.float32)
    img2[3:6, 3:6] = 1.0
    cases['small_square'] = img2
    
    # Cas 3: Image avec gradient
    img3 = np.zeros((7, 7), dtype=np.float32)
    img3[2:5, 2:5] = [[0.3, 0.5, 0.7],
                       [0.5, 1.0, 0.5],
                       [0.7, 0.5, 0.3]]
    cases['gradient'] = img3
    
    # Cas 4: Image plus grande
    img4 = np.zeros((15, 15), dtype=np.float32)
    img4[5:10, 5:10] = 1.0
    img4[7, 12] = 1.0
    cases['large_square'] = img4
    
    return cases

def create_odd_asymmetric_structuring_elements():
    """Crée plusieurs éléments structurants impairs non symétriques"""
    
    # Élément 3x3 en L
    se1 = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=np.float32)
    
    # Élément 3x3 diagonal
    se2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Élément 3x3 en escalier
    se3 = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.float32)
    
    # Élément 5x5 croix asymétrique
    se4 = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.float32)
    
    # Élément 5x5 diagonal étendu
    se5 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Élément 7x7 spiral
    se6 = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ], dtype=np.float32)
    
    # Élément 3x3 avec poids variables
    se7 = np.array([
        [0.5, 0.0, 0.0],
        [1.0, 1.5, 0.0],
        [0.5, 1.0, 2.0]
    ], dtype=np.float32)
    
    # Élément 5x5 avec poids variables
    se8 = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.5, 1.5, 0.5, 0.0],
        [1.0, 1.5, 2.0, 0.0, 0.0],
        [0.0, 0.5, 1.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    return {
        '3x3_L-shape': se1,
        '3x3_diagonal': se2,
        '3x3_stairs': se3,
        '5x5_asym_cross': se4,
        '5x5_diagonal_extended': se5,
        '7x7_spiral': se6,
        '3x3_weighted_L': se7,
        '5x5_weighted_cross': se8
    }

def detailed_comparison(image, structuring_element, operation='dilation'):
    """Compare en détail une opération morphologique"""
    
    print(f"\n{'='*80}")
    print(f"ANALYSE DÉTAILLÉE - {operation.upper()}")
    print(f"{'='*80}")
    print(f"Image shape: {image.shape}")
    print(f"Structuring element shape: {structuring_element.shape}")
    print(f"\nStructuring element:")
    print(structuring_element)
    
    # SCIPY
    if operation == 'dilation':
        scipy_result = ndimage.grey_dilation(image, structure=structuring_element, mode='constant', cval=-1000000)
    else:
        scipy_result = ndimage.grey_erosion(image, structure=structuring_element, mode='constant', cval=1000000)
    
    # MORPHOCORE
    image_torch = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    kernel_torch = torch.from_numpy(structuring_element).unsqueeze(0).unsqueeze(0)
    
    if operation == 'dilation':
        morphocore_result = dilation(image_torch, kernel_torch, channel_merge_mode="max")
    else:
        morphocore_result = erosion(image_torch, kernel_torch, channel_merge_mode="max")
    
    morphocore_result = morphocore_result.squeeze().numpy()
    
    # Calcul des différences
    diff = scipy_result - morphocore_result
    abs_diff = np.abs(diff)
    
    print(f"\n--- STATISTIQUES ---")
    print(f"Scipy   - min: {scipy_result.min():.4f}, max: {scipy_result.max():.4f}, mean: {scipy_result.mean():.4f}")
    print(f"Morpho  - min: {morphocore_result.min():.4f}, max: {morphocore_result.max():.4f}, mean: {morphocore_result.mean():.4f}")
    print(f"\nDifférence (Scipy - Morpho):")
    print(f"  MAE: {abs_diff.mean():.6f}")
    print(f"  MSE: {(diff**2).mean():.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6f}")
    print(f"  Min diff: {diff.min():.6f}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Pixels différents (>1e-6): {np.sum(abs_diff > 1e-6)}/{diff.size}")
    
    # Localiser les différences significatives
    significant_diff_mask = abs_diff > 1e-4
    if np.any(significant_diff_mask):
        print(f"\n--- LOCALISATION DES DIFFÉRENCES SIGNIFICATIVES (>1e-4) ---")
        diff_positions = np.argwhere(significant_diff_mask)
        print(f"Nombre de pixels avec différence significative: {len(diff_positions)}")
        print(f"\nPremiers pixels différents (max 10):")
        for i, (y, x) in enumerate(diff_positions[:10]):
            print(f"  Position ({y},{x}): Scipy={scipy_result[y,x]:.4f}, Morpho={morphocore_result[y,x]:.4f}, Diff={diff[y,x]:.4f}")
    else:
        print("\n✓ AUCUNE DIFFÉRENCE SIGNIFICATIVE - Les résultats matchent parfaitement!")
    
    return {
        'scipy': scipy_result,
        'morphocore': morphocore_result,
        'diff': diff,
        'abs_diff': abs_diff
    }

def visualize_detailed_comparison(image, results, se, operation, case_name):
    """Visualisation détaillée des résultats"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Image originale
    im0 = axes[0, 0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0, 0].set_title(f'Image Originale\n{case_name}')
    if image.shape[0] <= 9:  # N'affiche les valeurs que pour petites images
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                axes[0, 0].text(j, i, f'{image[i,j]:.2f}', ha='center', va='center', 
                              color='red' if image[i,j] > 0.5 else 'yellow', fontsize=8)
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Structuring element
    im1 = axes[0, 1].imshow(se, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title(f'Structuring Element\n{se.shape}')
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            axes[0, 1].text(j, i, f'{se[i,j]:.1f}', ha='center', va='center', 
                          color='white', fontsize=8, weight='bold')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Scipy result
    im2 = axes[0, 2].imshow(results['scipy'], cmap='gray', interpolation='nearest')
    axes[0, 2].set_title(f'{operation.capitalize()} (Scipy)')
    if results['scipy'].shape[0] <= 9:
        for i in range(results['scipy'].shape[0]):
            for j in range(results['scipy'].shape[1]):
                axes[0, 2].text(j, i, f'{results["scipy"][i,j]:.2f}', ha='center', va='center',
                              color='red' if results['scipy'][i,j] > 0.5 else 'yellow', fontsize=8)
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Morphocore result
    im3 = axes[1, 0].imshow(results['morphocore'], cmap='gray', interpolation='nearest')
    axes[1, 0].set_title(f'{operation.capitalize()} (Morphocore)')
    if results['morphocore'].shape[0] <= 9:
        for i in range(results['morphocore'].shape[0]):
            for j in range(results['morphocore'].shape[1]):
                axes[1, 0].text(j, i, f'{results["morphocore"][i,j]:.2f}', ha='center', va='center',
                              color='red' if results['morphocore'][i,j] > 0.5 else 'yellow', fontsize=8)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Différence absolue
    im4 = axes[1, 1].imshow(results['abs_diff'], cmap='hot', interpolation='nearest')
    axes[1, 1].set_title(f'Différence Absolue\n(MAE: {results["abs_diff"].mean():.6f})')
    if results['abs_diff'].shape[0] <= 9:
        for i in range(results['abs_diff'].shape[0]):
            for j in range(results['abs_diff'].shape[1]):
                if results['abs_diff'][i,j] > 1e-4:
                    axes[1, 1].text(j, i, f'{results["abs_diff"][i,j]:.3f}', ha='center', va='center',
                                  color='white', fontsize=7, weight='bold')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Différence signée
    max_diff = results['abs_diff'].max()
    im5 = axes[1, 2].imshow(results['diff'], cmap='RdBu_r', interpolation='nearest', 
                           vmin=-max_diff if max_diff > 0 else -1, 
                           vmax=max_diff if max_diff > 0 else 1)
    axes[1, 2].set_title(f'Différence (Scipy - Morpho)\nMax: {results["diff"].max():.4f}, Min: {results["diff"].min():.4f}')
    if results['diff'].shape[0] <= 9:
        for i in range(results['diff'].shape[0]):
            for j in range(results['diff'].shape[1]):
                if abs(results['diff'][i,j]) > 1e-4:
                    axes[1, 2].text(j, i, f'{results["diff"][i,j]:.3f}', ha='center', va='center',
                                  color='black', fontsize=7, weight='bold')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    filename = f'debug_{operation}_{case_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  → Sauvegardé: {filename}")
    plt.close()

def run_comprehensive_tests():
    """Lance une série de tests complets avec kernels impairs"""
    
    test_cases = create_simple_test_cases()
    structuring_elements = create_odd_asymmetric_structuring_elements()
    
    print("\n" + "#"*80)
    print("# TESTS COMPLETS AVEC KERNELS IMPAIRS UNIQUEMENT")
    print("#"*80)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for se_name, se in structuring_elements.items():
        print(f"\n{'='*80}")
        print(f"ÉLÉMENT STRUCTURANT: {se_name}")
        print(f"{'='*80}")
        
        for case_name, image in test_cases.items():
            print(f"\n--- Test: {se_name} + {case_name} ---")
            
            # Test dilation
            total_tests += 1
            results_dil = detailed_comparison(image, se, operation='dilation')
            visualize_detailed_comparison(image, results_dil, se, 'dilation', 
                                        f'{se_name}_{case_name}')
            
            if results_dil['abs_diff'].max() < 1e-4:
                passed_tests += 1
                print("  ✓ DILATION: PASS")
            else:
                failed_tests.append(f"DILATION: {se_name} + {case_name}")
                print("  ✗ DILATION: FAIL")
            
            # Test erosion
            total_tests += 1
            results_ero = detailed_comparison(image, se, operation='erosion')
            visualize_detailed_comparison(image, results_ero, se, 'erosion', 
                                        f'{se_name}_{case_name}')
            
            if results_ero['abs_diff'].max() < 1e-4:
                passed_tests += 1
                print("  ✓ EROSION: PASS")
            else:
                failed_tests.append(f"EROSION: {se_name} + {case_name}")
                print("  ✗ EROSION: FAIL")
    
    # Résumé final
    print("\n" + "#"*80)
    print("# RÉSUMÉ DES TESTS")
    print("#"*80)
    print(f"\nTotal tests: {total_tests}")
    print(f"Tests réussis: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Tests échoués: {len(failed_tests)}")
    
    if failed_tests:
        print("\n--- Tests échoués ---")
        for test in failed_tests:
            print(f"  ✗ {test}")
    else:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")

if __name__ == "__main__":
    run_comprehensive_tests()