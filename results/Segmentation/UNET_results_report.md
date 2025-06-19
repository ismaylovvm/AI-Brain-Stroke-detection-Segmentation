### Final Evaluation Metrics

- **F1 Score**: `0.8000`  

### Loss Graph (Left)

- **Train Loss** decreased significantly in early epochs, stabilizing around **epoch 10**.
- **Validation Loss** also followed a sharp drop, aligning closely with training loss near the end.
- Both curves indicate consistent optimization and minimal overfitting.

### Dice Coefficient Graph (Right)

- **Train Dice Coefficient** increased steadily, reaching nearly **0.8** at the final epoch.
- **Validation Dice Coefficient** followed a similar trend, with minor fluctuations.
- Indicates good generalization to unseen data.

## Conclusion

The model shows successful segmentation performance for both hemorrhagic and ischemic regions using UNET.  
F1 score of **0.8** demonstrates reliable stroke region detection capability, with further room for improvement through data augmentation or architectural tuning.

