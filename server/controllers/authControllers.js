const {User,Request} = require("../model/authModel");
const jwt = require("jsonwebtoken");


const createToken = (id) => {
  const maxAge = 3 * 24 * 60 * 60;
  return jwt.sign({ id }, "kishan sheth super secret key", {
    expiresIn: maxAge,
  });
};

const handleErrors = (err) => {
  let errors = { regis: "", password: "" };

  console.log(err);
  if (err.message === "incorrect Register No") {
    errors.regis = "Register Number is not Valid";
  }

  if (err.message === "incorrect password") {
    errors.password = "Password is incorrect";
  }

  if (err.message.includes("Users validation failed")) {
    Object.values(err.errors).forEach(({ properties }) => {
      errors[properties.path] = properties.message;
    });
  }

  return errors;
};

// const loo = async (regis, password) => {
//   const user = await User.findOne({ regis });
//   if (user) {
//     if (user.password===password) {
//       return user;
//     }
//     throw Error("incorrect password");
//   }
//   throw Error("incorrect Register No");
// };

// module.exports.login = async (req, res) => {
//   const { regis, password } = req.body;
//   try {
//     const user = loo(regis, password);
//     const token = createToken(user._id);
//     res.cookie("jwt", token, { httpOnly: false, maxAge: maxAge * 1000 });
//     res.status(200).json({ user: user._id, status: true });
//   } catch (err) {
//     const errors = handleErrors(err);
//     res.json({ errors, status: false });
//   }
// };

module.exports = {
  handleErrors,
  createToken
}