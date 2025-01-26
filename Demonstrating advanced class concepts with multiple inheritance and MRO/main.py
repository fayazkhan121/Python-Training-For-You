def print_mro(cls):
    print(f"\nMRO for {cls.__name__}:")
    for idx, class_ in enumerate(cls.__mro__, 1):
        print(f"{idx}. {class_.__name__}")

def main():
    # Create instances
    circle = HybridCircle(radius=5, color="blue", material="steel")
    
    # Demonstrate MRO
    print_mro(HybridCircle)
    
    # Demonstrate property access and method calls
    print(f"\nCircle Properties:")
    print(f"Color: {circle.color}")
    print(f"Area: {circle.area():.2f}")
    print(f"Mass: {circle.mass():.2f}")
    
    # Demonstrate dynamic property updates
    circle.color = "red"
    circle.radius = 7
    
    print(f"\nUpdated Circle Properties:")
    print(f"Color: {circle.color}")
    print(f"Area: {circle.area():.2f}")
    print(f"Mass: {circle.mass():.2f}")

if __name__ == "__main__":
    main()