class Human:
    def __init__(self, name="Norio"):
        self.name = name

    @property
    def nickname(self):
        return "_"+self.name+"_"

    @property
    def sayName(self):
        print("Hi {}".format(self.name))
        return "Hi {}".format(self.name)

if __name__ == "__main__":
    me = Human()
    print(me.nickname)
    me.sayName